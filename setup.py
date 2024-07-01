from setuptools import setup

package_name = 'sb3_her_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name,]),
        ('share/' + package_name, ['package.xml']),
        ('share/sb3_her_navigation/maps', ['resource/maps/map.txt',])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cris',
    maintainer_email='cris.lima.froes@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = sb3_her_navigation.train:main',
            'path_planning = sb3_her_navigation.path_planning:main',
        ],
    },
)
