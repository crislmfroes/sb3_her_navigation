# Install

Install dependencies with the following command:

```sh
pip install -r requirements.txt
```

Then build the colcon workspace.

# Run Path Planning

Launch stage with the following command:

```sh
ros2 launch stage_ros2 stage.launch.py world:=cave enforce_prefixes:=false one_tf_tree:=true
```

Then run path planning node with following command:

```sh
ros2 run sb3_her_navigation path_planning
```

After some time the robot will start to navigate to the waypoints.