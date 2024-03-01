# pipeline
This folder contains the code needed for performing the full bin-picking pipeline as indicated in the paper.

## moveit2_ws-moveit_code
Contains the moveit code needed for path planning

## ros2_ws-roscode
Contains the code to create the collision models of detected objects.

# Structure of project
There are two workspaces: a workspace for ROS2 - `ros2_ws` -, where the implementation is done in Python; a workspace for Moveit2 - `ws_moveit2` -, which is implemented in c++. <br>

- Ubuntu 22.04, with ROS2 Humble and Moveit2 + [Task Constructor](https://moveit.picknik.ai/humble/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html) (just follow the manual to install).

/home <br>
- /ros2_ws <br>
- - /build <br>
- - /install <br>
- - - /percept_wmodel <br>
- - - - /lib <br>
- - - - /share <br>
- - - - - /percept_wmodel <br>
- - - - - - /launch <br>
- - /log <br>
- - /src <br>
- - - /ros_tutorials <br>
- - - /percept_wmodel <br>
- - - - /resource <br>
- - - - /percept_wmodel <br>
- - - - - 60_v46.torch <br>
- - - - - add_frame.py <br>
- - - - - /_/_init/_/_.py <br>
- - - - - instance_seg_realsense.py <br>
- - - - - pipeline_settings.txt <br>
- - - - - segmentation.py <br>
- - - - - listener_to_csv.py <br>
- - - - - imseg_to_cobj.py <br>
- - - - /test <br>
- - - - setup.cfg <br>
- - - - setup.py <br>
- - - - package.xml<br>
- /ws_moveit2 <br>
- - /build<br>
- - /install<br>
- - - /mtc_tutorial<br>
- - - - /share<br>
- - - - - /mtc_tutorial<br>
- - - - - - /launch<br>
- - - - - - - pick_place_demo.launch.py<br>
- - - /rs_move<br>
- - - - /share<br>
- - - - - /rs_move<br>
- - - - - - /launch<br>
- - - - - - - launch_rviz.launch.py<br>
- - - - - - - move_group_interface_percept_wmodel.launch.py<br>
- - /log<br>
- - /src<br>
- - - /mtc_tutorial<br>
- - - /rs_move<br>
- - - - /include<br>
- - - - /src<br>
- - - - - csv.h<br>
- - - - - move_group_interface_percept_wmodel.cpp<br>
- - - - - out.csv<br>
- - - - - param_node.cpp<br>
- - - - - read_csv2.cpp<br>
- - - - - test_publisher.cpp<br>
- - - - CMakeLists.txt<br>
- - - - package.xml<br>
