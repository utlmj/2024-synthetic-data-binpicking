// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_position_example_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

//// added for csv stuff \/
#include "csv.h" // from https://github.com/ben-strasser/fast-cpp-csv-parser// should be in same folder as this code
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <franka_gripper/MoveAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <franka_gripper/GraspAction.h>


// N is the number of datapoints in the trajectory csv
int N = 276801;
double pos[276801][9];
double t_traj[276801];
bool run = true;

int k = -1;
int g = 0;

std::ofstream time_out;

bool closed = false;
bool open = true;

double sample_rate;

actionlib::SimpleActionClient<franka_gripper::GraspAction> ac("franka_gripper/grasp", true);
franka_gripper::GraspGoal goal;

namespace franka_example_controllers {

bool JointPositionExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointPositionExampleController: Error getting position joint interface from hardware!");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointPositionExampleController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointPositionExampleController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  position_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointPositionExampleController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}}; // STARTPOS
  for (size_t i = 0; i < q_start.size(); i++) {
    if (std::abs(position_joint_handles_[i].getPosition() - q_start[i]) > 0.1) {
      ROS_ERROR_STREAM(
          "JointPositionExampleController: Robot is not in the expected starting position for "
          "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
          "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
      return false;
    }
  }


  /// load position data
  int i = 0;

  io::CSVReader<10> in("joint_pos_path_2_resampled_0.0001.csv");//("successtest.csv");//"out.csv");
  in.read_header(
  io::ignore_extra_column,
    "time","panda_joint1",
  "panda_joint2",	"panda_joint4",
  "panda_joint5",	"panda_joint6",
  "panda_joint7",	"panda_finger_joint1",
  "panda_joint3",	"panda_finger_joint2"
  );

  double time_s; double time_ns; 
  double panda_joint1; 
  double panda_joint2; double panda_joint4; 
  double panda_joint5; double panda_joint6;
  double panda_joint7; double panda_finger_joint1;
  double panda_joint3; double panda_finger_joint2;     
  while(in.read_row(time_s,panda_joint1,panda_joint2,panda_joint4,panda_joint5,
    panda_joint6,panda_joint7, panda_finger_joint1, panda_joint3, panda_finger_joint2)){
    i++;

    double delta_angle = 1;// M_PI;
    pos[i][0] = delta_angle*panda_joint1;
    pos[i][1] = delta_angle*panda_joint2;
    pos[i][2] = delta_angle*panda_joint3;
    pos[i][3] = delta_angle*panda_joint4;
    pos[i][4] = delta_angle*panda_joint5;
    pos[i][5] = delta_angle*panda_joint6;
    pos[i][6] = delta_angle*panda_joint7;
    pos[i][7] = panda_finger_joint1;
    pos[i][8] = panda_finger_joint2;
    t_traj[i] = time_s;
    std::cout<<i; std::cout<<", ";
    }

    // ROS_INFO("Waiting for action server to start.");

    ac.waitForServer();

    // ROS_INFO("Action server started, sending goal.");

    time_out.open("time_out.csv");

    sample_rate = t_traj[4]-t_traj[3];
    std::cout<<"sample_rate: "; std::cout<<sample_rate;

    int h = system("rosrun franka_gripper homing_gripper"); // home the gripper


  return true;
}


void JointPositionExampleController::starting(const ros::Time& /* time */) {
  for (size_t i = 0; i < 7; ++i) {
    initial_pose_[i] = position_joint_handles_[i].getPosition();
    std::cout<<initial_pose_[i]; std::cout<<", ";
  }
  elapsed_time_ = ros::Duration(0.0);
}

void JointPositionExampleController::update(const ros::Time& /*time*/,
                                            const ros::Duration& period) {
  elapsed_time_ += period;
  double timestep = period.toSec();
  if (k<N-2){
    int c;
    c = round(timestep*1000);

    // this also works for different sampling rates in the trajectory data
    // c = round(timestep/sample_rate); //e.g. 0.001/0.0001 = 10; 
    // std::cout<<c; std::cout<<", ";
    k+=c;

    
  }
  if(run){
    // std::cout<<" k: "; std::cout<<k;
      for (int n = 0; n < 7; n++) {
        position_joint_handles_[n].setCommand(pos[k][n]);      
      }
      //             //smaller than the commanded grasp width.
      // // 
      if(k >= 20000&&open == true){
        open = false;
      // /////////CLOSE GRIPPER///////////
      goal.width = pos[k][7] + pos[k][8];//+0.04; 
      goal.speed = 0.2;
      goal.force = 5.0;
      goal.epsilon.inner = 0.001;     // Maximum tolerated deviation when the actual grasped width is
      goal.epsilon.outer = 0.001;     // Maximum tolerated deviation when the actual grasped width is
      // //                           // larger than the commanded grasp width.
      ac.sendGoal(goal);              // Sending the Grasp command to gripper
      std::cout<<goal.width; std::cout<<", ";
    
        
      }
      
    }

    time_out << std::to_string(elapsed_time_.toSec());
    time_out << "\n";
  
  std::cout<<k; std::cout<<", ";
  }

}

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointPositionExampleController,
                       controller_interface::ControllerBase)
