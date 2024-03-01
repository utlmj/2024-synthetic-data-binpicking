#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/MoveAction.h>

int main (int argc, char **argv)
{
    ros::init(argc, argv, "open_gripper");

    actionlib::SimpleActionClient<franka_gripper::MoveAction> ac_open("franka_gripper/move", true);

    franka_gripper::MoveGoal goal_open; 
    // float64 width  # [m]
    // float64 speed  # [m/s]

    goal_open.width = 0.1;
    goal_open.speed = 0.1;
    ROS_INFO("Waiting for action server to start.");

    ac_open.waitForServer();

    ROS_INFO("Action server started, sending goal.");
    
    ac_open.sendGoal(goal_open);          // Sending the Grasp command to gripper


    bool finished_before_timeout = ac_open.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState op_state = ac_open.getState();
        ROS_INFO("Action finished: %s.", op_state.toString().c_str());
    }
    else
    ROS_INFO("Action did not finish before the time out.");

    //exit
    return 0;
}
