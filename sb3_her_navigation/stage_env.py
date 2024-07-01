from gymnasium import Env
from gymnasium.spaces import Box, Dict
import numpy as np
import rclpy
import ros2launch
from rclpy.node import Node
from tf2_ros.transform_listener import TransformListener, Buffer
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from std_srvs.srv import Empty
from transforms3d.euler import quat2euler, euler2quat
from threading import Event
import os
import math
import networkx as nx
from copy import deepcopy
from threading import Thread
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.sac.sac import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from gymnasium.wrappers import TimeLimit
import cv2
from ament_index_python.packages import get_package_share_path


class StageEnv(Node, Env):
    def __init__(self):
        super().__init__('stage_env')
        self.observation_space = Dict({
            'observation': Box(low=-np.inf, high=np.inf, shape=(273,)),
            'achieved_goal': Box(low=-np.inf, high=np.inf, shape=(2,)),
            'desired_goal': Box(low=-np.inf, high=np.inf, shape=(2,)),
        })
        self.action_space = Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,))
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 0)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 0)
        self.odom_sub = self.create_subscription(Odometry, '/ground_truth', self._update_odometry, 0)
        self.laser_sub = self.create_subscription(LaserScan, '/base_scan', self._update_scan, 0)
        self.reset_proxy = self.create_client(Empty, '/reset_positions')
        self.trigger = self.create_service(Empty, '/execute_task', self.execute_task)
        self.save_map = self.create_service(Empty, '/save_map', self.save_map)
        self.train_service = self.create_service(Empty, '/train', self.train)
        self.control_rate = self.create_rate(1000)
        self.grid_resolution = 0.1
        self.map_side = 20.0
        self.default_kernel_size = 5
        self.kernel_size = self.default_kernel_size
        self.control_horizon = math.inf
        self.map_frame = "robot_0/odom"
        self.map_path = os.path.join(get_package_share_path('sb3_her_navigation'), 'maps/map.txt')
        try:
            self.grid = np.loadtxt(self.map_path)
            assert self.grid.shape == (int(self.map_side//self.grid_resolution), int(self.map_side//self.grid_resolution))
            self.grid = (cv2.dilate(self.grid*255, cv2.getGaussianKernel(ksize=15, sigma=1.0)) > 0) * 1
            map_msg = OccupancyGrid()
            map_info = MapMetaData()
            map_info.resolution = self.grid_resolution
            map_info.height = int(self.map_side//self.grid_resolution)
            map_info.width = int(self.map_side//self.grid_resolution)
            map_info.origin.position.x = 0.0
            map_info.origin.position.y = 0.0
            quat = euler2quat(0.0, 0.0, 0.0)
            map_info.origin.orientation.w = quat[0]
            map_info.origin.orientation.x = quat[1]
            map_info.origin.orientation.y = quat[2]
            map_info.origin.orientation.z = quat[3]
            map_msg.info = map_info
            map_msg.data = (self.grid.T.flatten().astype(int)*100).tolist()
            map_msg.header.frame_id = self.map_frame
            self.map_publisher.publish(map_msg)
        except FileNotFoundError:
            self.grid = np.zeros(shape=(int(self.map_side//self.grid_resolution), int(self.map_side//self.grid_resolution)))
            self.tf_timer = self.create_timer(1/30.0, self.update_gridmap)
        self.buffer = Buffer(node=self)
        self.tfl = TransformListener(buffer=self.buffer, node=self)
        self.scan = None
        self.odometry = None
        self.set_goal(*self.world2odom(0.0, 0.0))

    def save_map(self, req, res):
        np.savetxt(self.map_path, self.grid)
        return res

    def execute_task(self, req, res):
        def run_sync():
            goals = [
                self.world2odom(5.0, 4.0),
                self.world2odom(4.0, -4.0)
            ]
            for goal in goals:
                self.set_goal(*goal)
                self.navigate_to_goal()
        run_sync()
        return res

    def train(self, req, res):
        def run_async():
            env = TimeLimit(self, max_episode_steps=1000)
            trainer = SAC(policy='MultiInputPolicy', env=env, replay_buffer_class=HerReplayBuffer, learning_starts=10000, verbose=1)
            trainer.learn(total_timesteps=100000)
            trainer.policy.save(path='sb3_sac_her_stage')
        run_async()
        return res

    def coord2gridmap(self, x, y):
        cell_x = (x - (-10.0))//self.grid_resolution
        cell_y = (y - (-10.0))//self.grid_resolution
        return int(cell_x), int(cell_y)

    def gridmap2coord(self, cell_x, cell_y):
        x = (cell_x*self.grid_resolution) + (-10.0)
        y = (cell_y*self.grid_resolution) + (-10.0)
        return x, y

    def world2odom(self, x, y):
        return x, y

    def update_gridmap(self):
        try:
            if self.scan == None:
                return
            scan = self.scan
            points = []
            for i, r in enumerate(scan.ranges):
                if r > 4.5:
                    continue
                angle = scan.angle_min + i*scan.angle_increment
                center_x, center_y, center_theta = self._get_obs()['observation'][:3]
                x = r*math.cos(angle + center_theta) + center_x
                y = r*math.sin(angle + center_theta) + center_y
                ps = PointStamped()
                ps.point.x = x
                ps.point.y = y
                ps.header.frame_id = self.map_frame

                points.append(ps)
            for ps in points:
                x = ps.point.x
                y = ps.point.y
                cell_x, cell_y = self.coord2gridmap(x, y)
                center_x, center_y, center_theta = self._get_obs()['observation'][:3]
                interp_points = np.linspace(start=np.array([center_x, center_y]), stop=np.array([x, y]))
                for i_p in interp_points:
                    interp_cell_x, interp_cell_y = self.coord2gridmap(*i_p)
                    self.grid[interp_cell_x, interp_cell_y] = 0
                self.grid[cell_x, cell_y] = 1
                lower = int(-self.kernel_size//2)
                higher = int(self.kernel_size//2)
                for k_i in range(lower, higher):
                    for k_j in range(lower, higher):
                        try:
                            self.grid[cell_x+k_i, cell_y+k_j] = 1
                        except:
                            pass
        except BaseException as e:
            self.get_logger().error(str(e))
            raise(e)
        map_msg = OccupancyGrid()
        map_info = MapMetaData()
        map_info.resolution = self.grid_resolution
        map_info.height = int(self.map_side//self.grid_resolution)
        map_info.width = int(self.map_side//self.grid_resolution)
        map_info.origin.position.x = 0.0
        map_info.origin.position.y = 0.0
        quat = euler2quat(0.0, 0.0, 0.0)
        map_info.origin.orientation.w = quat[0]
        map_info.origin.orientation.x = quat[1]
        map_info.origin.orientation.y = quat[2]
        map_info.origin.orientation.z = quat[3]
        map_msg.info = map_info
        map_msg.data = (self.grid.T.flatten().astype(int)*100).tolist()
        map_msg.header.frame_id = self.map_frame
        self.map_publisher.publish(map_msg)

    def hash_grid_cell(self, cell_x, cell_y):
        return f'{cell_x};{cell_y}'

    def unhash_grid_cell(self, hash):
        return [float(e) for e in hash.split(';')]

    def get_path(self):
        pose_graph = nx.Graph()
        all_poses = np.zeros(shape=(self.grid.shape[0]*self.grid.shape[1], 2))
        all_poses_mask = np.zeros(shape=(self.grid.shape[0]*self.grid.shape[1],), dtype=bool)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                all_poses[i*self.grid.shape[0]+(j%self.grid.shape[1])] = self.gridmap2coord(i, j)
                all_poses_mask[i*self.grid.shape[0]+(j%self.grid.shape[1])] = self.grid[i,j] == 1
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i,j] == 1:
                    continue
                curr_hash = self.hash_grid_cell(i, j)
                for step_i in [-1,0,1]:
                    for step_j in [-1,0,1]:
                        neigh_i = i + step_i
                        neigh_j = j + step_j
                        neigh_i = max(neigh_i, 0)
                        neigh_i = min(neigh_i, self.map_side//self.grid_resolution - 1)
                        neigh_j = max(neigh_j, 0)
                        neigh_j = min(neigh_j, self.map_side//self.grid_resolution - 1)
                        neigh_i = int(neigh_i)
                        neigh_j = int(neigh_j)
                        if self.grid[neigh_i, neigh_j] == 1:
                            continue
                        neigh_hash = self.hash_grid_cell(neigh_i, neigh_j)
                        start_i = max(i - 10, 0)
                        stop_i = min(i + 10, self.grid.shape[0]-1)
                        start_j = max(j - 10, 0)
                        stop_j = min(j + 10, self.grid.shape[1]-1)
                        pose_graph.add_edge(curr_hash, neigh_hash, weight=sum(self.grid[start_i:stop_i, start_j:stop_j].flatten()))
        all_nodes = sorted(pose_graph.nodes, key=lambda h: np.linalg.norm(np.array(self.gridmap2coord(*self.unhash_grid_cell(h))) - self.goal))
        target = all_nodes[0]
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y
        all_nodes = sorted(pose_graph.nodes, key=lambda h: np.linalg.norm(np.array(self.gridmap2coord(*self.unhash_grid_cell(h))) - np.array([x, y])))
        source = all_nodes[0]
        path = nx.astar_path(pose_graph, source, target)
        return [self.gridmap2coord(*self.unhash_grid_cell(h)) for h in path]

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -1.0 * (np.linalg.norm(achieved_goal - desired_goal, axis=-1) > 0.15)
    
    def _update_odometry(self, msg: Odometry):
        self.odometry = deepcopy(msg)

    def _update_scan(self, msg: LaserScan):
        self.scan = deepcopy(msg)

    def set_goal(self, x, y):
        self.goal = np.array([x, y])

    def normalize_angle(self, theta):
        if theta < 0:
                theta = theta % (-2*math.pi)
        else:
            theta = theta % (2*math.pi)
        return theta

    def _get_obs(self):
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y
        quat = [
            self.odometry.pose.pose.orientation.w,
            self.odometry.pose.pose.orientation.x,
            self.odometry.pose.pose.orientation.y,
            self.odometry.pose.pose.orientation.z,
        ]
        _, _, theta = quat2euler(quat)
        ranges = self.scan.ranges
        return {
            'observation': np.array([x, y, theta, *ranges]),
            'achieved_goal': np.array([x, y]),
            'desired_goal': self.goal,
        }

    def check_collision(self):
        return min(self._get_obs()['observation'][3:]) < 0.5

    def reset(self, **kwargs):
        future = self.reset_proxy.call(Empty.Request())
        lower = np.array([-8.0, -8.0])
        higher = np.array([8.0, 8.0])
        goal = (higher - lower) * np.random.random_sample(size=(2,)) + lower
        self.set_goal(*goal)
        obs =  self._get_obs()
        return obs, {}

    def step(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        self.vel_publisher.publish(cmd_vel)
        self.control_rate.sleep()
        obs = self._get_obs()
        info = {}
        reward = self.compute_reward(achieved_goal=obs['achieved_goal'], desired_goal=obs['desired_goal'], info=info)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def rotate(self, angle):
        if abs(angle - math.radians(270)) <= 0.05:
            angle = math.radians(-90.0)
        if abs(angle - math.radians(-270)) <= 0.05:
            angle = math.radians(90.0)
        _, _, start_theta = self._get_obs()['observation'][:3]
        theta = start_theta
        error = self.normalize_angle(self.normalize_angle(theta - start_theta) - self.normalize_angle(angle))
        kp = 1.0
        while abs(error) > 0.5:
            self.step(np.array([0.0, -kp*error]))
            _, _, theta = self._get_obs()['observation'][:3]
            error = self.normalize_angle(self.normalize_angle(theta - start_theta) - self.normalize_angle(angle))
        self.step(np.array([0.0, 0.0]))

    def rotate_absolute(self, angle):
        _, _, start_theta = self._get_obs()['observation'][:3]
        theta = start_theta
        error = theta - angle
        kp = 1.0
        if abs(theta) >= math.radians(135):
            thresh = math.pi/2
        else:
            thresh = 0.5
        while abs(error) > thresh:
            self.step(np.array([0.0, -kp*error]))
            _, _, theta = self._get_obs()['observation'][:3]
            error = theta - angle
        self.step(np.array([0.0, 0.0]))

    def move(self, distance):
        start_x, start_y, start_theta = self._get_obs()['observation'][:3]
        x = start_x
        y = start_y
        error = np.linalg.norm([x- start_x, y - start_y]) - distance
        kp = 10.0
        while abs(error) > 0.3:
            self.step(np.array([-kp*error, 0.0]))
            x, y, theta = self._get_obs()['observation'][:3]
            error = np.linalg.norm([x- start_x, y - start_y]) - distance
        self.step(np.array([0.0, 0.0]))

    def navigate_to_goal(self):
        arrived = False
        while not arrived:
            path = self.get_path()[1:]
            for i in range(min(len(path), self.control_horizon)):
                    next_x, next_y = path[i]
                    x, y, theta = self._get_obs()['observation'][:3]
                    next_x, next_y = path[i]
                    angle1 = math.atan2(next_y-y, next_x-x) - theta
                    angle2 = theta - math.atan2(next_y-y, next_x-x)
                    angles = [angle1, angle2]
                    angles = sorted(angles, key=lambda a: abs(self.normalize_angle(a)))
                    angle = angles[0]
                    angle = min(math.pi - 1e-5, angle)
                    angle = max(-math.pi + 1e-5, angle)
                    self.rotate(angles[0])
                    x, y, theta = self._get_obs()['observation'][:3]
                    distance = np.linalg.norm(np.array([next_x-x, next_y-y]))
                    self.move(distance)
            arrived = True