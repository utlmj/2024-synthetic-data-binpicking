#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/MoveAction.h>

int main (int argc, char **argv)
{
    ros::init(argc, argv, "close_gripper");

    actionlib::SimpleActionClient<franka_gripper::MoveAction> ac_close("franka_gripper/move", true);

    franka_gripper::MoveGoal goal_close; 
    // float64 width  # [m]
    // float64 speed  # [m/s]

    goal_close.width = 0.04;
    goal_close.speed = 0.7;
    ROS_INFO("Waiting for action server to start.");

    ac_close.waitForServer();

    ROS_INFO("Action server started, sending goal.");
    
    ac_close.sendGoal(goal_close);          // Sending the Grasp command to gripper


    bool finished_before_timeout = ac_close.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState cl_state = ac_close.getState();
        ROS_INFO("Action finished: %s.", cl_state.toString().c_str());
    }
    else
    ROS_INFO("Action did not finish before the time out.");

    //exit
    return 0;
}
