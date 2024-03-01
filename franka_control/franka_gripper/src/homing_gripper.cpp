#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/HomingAction.h>

int main (int argc, char **argv)
{
    ros::init(argc, argv, "homing_gripper");

    actionlib::SimpleActionClient<franka_gripper::HomingAction> ac("franka_gripper/homing", true);

    franka_gripper::HomingGoal goal;
    ROS_INFO("Waiting for action server to start.");

    ac.waitForServer();

    ROS_INFO("Action server started, sending goal.");

    ac.sendGoal(goal);          // Sending the Grasp command to gripper


    bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState state = ac.getState();
        ROS_INFO("Action finished: %s.", state.toString().c_str());
    }
    else
    ROS_INFO("Action did not finish before the time out.");

    //exit
    return 0;
}