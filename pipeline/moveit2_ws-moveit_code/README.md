# moveit2_ws-moveit_code
First, install MoveIt2 and [Moveit Task Constructor](https://moveit.picknik.ai/humble/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)
If you follow link above, you will create a package called mtc_tutorial. The package `mtc_tutorial` in our repository is the same except we edited 
the `mtc_tutorial.cpp` file. If you rename this file keep in mind to also rename change it in the `CMakeLists.txt`.

Create an extra package rs_move
The most important file of this package is the `launch_rviz.launch.py` file which launches rviz with the Panda robot, of which the
configuration file is in the launch folder of moveit2_tutorials


## Run the pipeline  
### in one terminal:  
`>> ros2 launch rs_move launch_rviz.launch.py`   
This launches rviz and camera frame (same as `mtc_demo.launch.py` but now with addframe node)

### in another terminal:  
`>> ros2 run percept_wmodel imseg_to_cobj`

#### in yet another terminal, after imseg_to_cobj is publishing the collisiondata:  
`>> ros2 launch mtc_tutorial pick_place_demo.launch.py`
This launches `mtc_tutorial.cpp`.
