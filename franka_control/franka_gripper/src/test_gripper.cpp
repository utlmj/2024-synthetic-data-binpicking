// #include <ros/ros.h>
// #include <actionlib/client/simple_action_client.h>
// #include <actionlib/client/terminal_state.h>
// #include <franka_gripper/MoveAction.h>

// int main (int argc, char **argv)
// {
//     ros::init(argc, argv, "test_gripper");

//     actionlib::SimpleActionClient<franka_gripper::MoveAction> ac("franka_gripper/move", true);

//     franka_gripper::MoveGoal goal; 
//     // float64 width  # [m]
//     // float64 speed  # [m/s]

//     goal.width = 0.004;
//     goal.speed = 0.1;
//     ROS_INFO("Waiting for action server to start.");

//     ac.waitForServer();

//     ROS_INFO("Action server started, sending goal.");
    
//     ac.sendGoal(goal);          // Sending the Grasp command to gripper


//     bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

//     if (finished_before_timeout)
//     {
//         actionlib::SimpleClientGoalState state = ac.getState();
//         ROS_INFO("Action finished: %s.", state.toString().c_str());
//     }
//     else
//     ROS_INFO("Action did not finish before the time out.");

//     //exit
//     return 0;
// }

//////////////////////////////HOMING////////////////////////////////////////
// #include <ros/ros.h>
// #include <actionlib/client/simple_action_client.h>
// #include <actionlib/client/terminal_state.h>
// #include <franka_gripper/HomingAction.h>

// int main (int argc, char **argv)
// {
//     ros::init(argc, argv, "test_gripper");

//     actionlib::SimpleActionClient<franka_gripper::HomingAction> ac("franka_gripper/homing", true);

//     franka_gripper::HomingGoal goal;
//     ROS_INFO("Waiting for action server to start.");

//     ac.waitForServer();

//     ROS_INFO("Action server started, sending goal.");

//     ac.sendGoal(goal);          // Sending the Grasp command to gripper


//     bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

//     if (finished_before_timeout)
//     {
//         actionlib::SimpleClientGoalState state = ac.getState();
//         ROS_INFO("Action finished: %s.", state.toString().c_str());
//     }
//     else
//     ROS_INFO("Action did not finish before the time out.");

//     //exit
//     return 0;
// }

////////////////////////////////////GRASPING///////////////////////////////////////////

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/GraspAction.h>
#include <franka_gripper/HomingAction.h>

int main (int argc, char **argv)
{
    ros::init(argc, argv, "test_gripper");
    

    actionlib::SimpleActionClient<franka_gripper::GraspAction> ac_test("franka_gripper/grasp", true);

  
    ROS_INFO("Waiting for action server to start.");

    ac_test.waitForServer();

    ROS_INFO("Action server started, sending goal.");

    franka_gripper::GraspGoal goal;
    goal.width = 0.06;     // [m]
    goal.speed = 0.07;           //  Closing speed. [m/s]
    goal.force = 0.5;            //   Grasping (continuous) force [N]
    goal.epsilon.inner = 0.01;  // Maximum tolerated deviation when the actual grasped width is
                                //smaller than the commanded grasp width.
    goal.epsilon.outer = 0.02;  // Maximum tolerated deviation when the actual grasped width is
                                // larger than the commanded grasp width.
    ac_test.sendGoal(goal);          // Sending the Grasp command to gripper


    bool finished_before_timeout = ac_test.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState gr_state = ac_test.getState();
        ROS_INFO("Action finished: %s.", gr_state.toString().c_str());
    }
    else
    ROS_INFO("Action did not finish before the time out.");

    //exit
    return 0;
}
