#define BOOST_BIND_NO_PLACEHOLDERS

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

#include "rclcpp/rclcpp.hpp"
#include <std_msgs/msg/float32_multi_array.hpp>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <chrono>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("mtc_tutorial");
namespace mtc = moveit::task_constructor;


// https://moveit.picknik.ai/humble/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html
/*not part of tutorial: start*/


using namespace std::chrono_literals;
using std::placeholders::_1;

// global variables
shape_msgs::msg::Mesh mesh_var;       //
moveit_msgs::msg::CollisionObject oc;
moveit_msgs::msg::CollisionObject oc_box;
moveit_msgs::msg::CollisionObject oc_placebox;
moveit_msgs::msg::CollisionObject oc_scene;
// std_msgs::msg::Float32MultiArray offset;
//moveit_msgs::msg::Grasp grasp;
int test_var;

// NODE FOR SUBSCRIBING TO COLLISION OBJECT MESSAGE
class readco : public rclcpp::Node
{
  public:

    readco()
    : Node("READCO") // name of the subscriber
    {
    	subscription_ = this->create_subscription<moveit_msgs::msg::CollisionObject>(
      	"CO", 10, std::bind(&readco::topic_callback, this, _1)); 
		
    }

  	private:
    void topic_callback(const moveit_msgs::msg::CollisionObject::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard a CHICKEN!");
	  test_var ++;
      oc.meshes=msg->meshes;
      oc.mesh_poses=msg->mesh_poses;
      oc.operation=msg->operation;
      oc.header.frame_id=msg->header.frame_id;
      oc.id=msg->id;
    }
    rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr subscription_;

	
};

class readscene : public rclcpp::Node
{
  public:

    readscene()
    : Node("READSCENE") // name of the subscriber
    {
    	subscription_ = this->create_subscription<moveit_msgs::msg::CollisionObject>(
      	"CO_SCENE", 10, std::bind(&readscene::topic_callback, this, _1)); 
		
    }

  	private:
    void topic_callback(const moveit_msgs::msg::CollisionObject::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard a SCENE!");
	  test_var ++;
      oc_scene.meshes=msg->meshes;
      oc_scene.mesh_poses=msg->mesh_poses;
      oc_scene.operation=msg->operation;
      oc_scene.header.frame_id=msg->header.frame_id;
      oc_scene.id=msg->id;
    }
    rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr subscription_;
};



// NODE FOR SUBSCRIBING TO COLLISION OBJECT MESSAGE
class readbox : public rclcpp::Node
{
  public:

    readbox()
    : Node("READBOX") // name of the subscriber
    {
    	subscriptionbox_ = this->create_subscription<moveit_msgs::msg::CollisionObject>(
      	"CO_BOX", 10, std::bind(&readbox::topic_callback_box, this, _1)); 
    }

  	private:
    void topic_callback_box(const moveit_msgs::msg::CollisionObject::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard a BOX!");
      oc_box.primitives=msg->primitives;
      oc_box.primitive_poses=msg->primitive_poses;
      oc_box.operation=msg->operation;
      oc_box.header.frame_id=msg->header.frame_id;
      oc_box.id=msg->id;
    }
    rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr subscriptionbox_;

	
};



// NODE FOR SUBSCRIBING TO COLLISION OBJECT MESSAGE
class readplacebox : public rclcpp::Node
{
  public:

    readplacebox()
    : Node("READPLACEBOX") // name of the subscriber
    {
    	subscriptionplacebox_ = this->create_subscription<moveit_msgs::msg::CollisionObject>(
      	"CO_PLACEBOX", 10, std::bind(&readplacebox::topic_callback_placebox, this, _1)); 
		
    }

  	private:
    void topic_callback_placebox(const moveit_msgs::msg::CollisionObject::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard a PLACEBOX!");
      oc_placebox.primitives=msg->primitives;
      oc_placebox.primitive_poses=msg->primitive_poses;
      oc_placebox.operation=msg->operation;
      oc_placebox.header.frame_id=msg->header.frame_id;
      oc_placebox.id=msg->id;
    }
    rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr subscriptionplacebox_;

	
};


/* not part of tutorial: end*/

class MTCTaskNode
{
public:
  MTCTaskNode(const rclcpp::NodeOptions& options);

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

  void doTask();

  void setupPlanningScene();

private:
  // Compose an MTC task from a series of stages.
  mtc::Task createTask();
  mtc::Task task_;
  rclcpp::Node::SharedPtr node_;
};

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface()
{
  return node_->get_node_base_interface();
}

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("mtc_node", options) }
{
}

/* BELOW SHOULD BE RECEIVING THE COLLISION OBJECT FROM THE COLLISION OBJECT PUBLISHER NODE*/
void MTCTaskNode::setupPlanningScene()
{

  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

  collision_objects.push_back(oc_scene);

  moveit::planning_interface::PlanningSceneInterface psi;
  psi.addCollisionObjects(collision_objects);

  /* chicken collision object from subscriber node: start*/
  psi.applyCollisionObject(oc);

}

void MTCTaskNode::doTask()
{
  
  task_ = createTask();

  try
  {
    task_.init();
  }
  catch (mtc::InitStageException& e)
  {
    RCLCPP_ERROR_STREAM(LOGGER, e);
    return;
  }

  if (!task_.plan(1))
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
    return;
  }
  task_.introspection().publishSolution(*task_.solutions().front());

//////////////////////////////////// BELOW IS THE EXECUTION PART, 
  auto result = task_.execute(*task_.solutions().front());
  if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
    return;
  }

  return;
}

mtc::Task MTCTaskNode::createTask()
{
  mtc::Task task;
  task.stages()->setName("demo task");
  task.loadRobotModel(node_);

  const auto& arm_group_name = "panda_arm";
  const auto& hand_group_name = "hand";
  const auto& hand_frame = "panda_hand";

  // Set task properties
  task.setProperty("group", arm_group_name);
  task.setProperty("eef", hand_group_name);
  task.setProperty("ik_frame", hand_frame);

// Disable warnings for this line, as it's a variable that's set but not used in this example
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
  mtc::Stage* current_state_ptr = nullptr;  // Forward current_state on to grasp pose generator
#pragma GCC diagnostic pop

  auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
  current_state_ptr = stage_state_current.get();
  task.add(std::move(stage_state_current));

  auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
  auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

  auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian_planner->setMaxVelocityScaling(0.8);
  cartesian_planner->setMaxAccelerationScaling(0.6);
  cartesian_planner->setStepSize(.001);

{
  auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
  stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
  stage->setGoal("ready");
  task.add(std::move(stage));
}

  auto stage_open_hand =
      std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
  stage_open_hand->setGroup(hand_group_name);
  stage_open_hand->setGoal("open");
  task.add(std::move(stage_open_hand));


  /*add extra stages*/
  
  auto stage_move_to_pick = std::make_unique<mtc::stages::Connect>(
      "move to pick",
      mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
  stage_move_to_pick->setTimeout(8.0);
  stage_move_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
  task.add(std::move(stage_move_to_pick));


  mtc::Stage* attach_object_stage =
      nullptr;  // Forward attach_object_stage to place pose generator

  {
    auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
    task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
    grasp->properties().configureInitFrom(mtc::Stage::PARENT,
                                          { "eef", "group", "ik_frame" });

      {
        auto stage =
            std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner);
        stage->properties().set("marker_ns", "approach_object");
        stage->properties().set("link", hand_frame);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
        stage->setMinMaxDistance(0.0, 0.15);

        // Set hand forward direction
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = hand_frame;
        vec.vector.x = 1.0;
        vec.vector.y = 1.0;
        vec.vector.z = -1.0;
        
        
        stage->setDirection(vec);
        grasp->insert(std::move(stage));
      }                               

      {
        // Sample grasp pose
        auto stage = std::make_unique<mtc::stages::GenerateGraspPose>("generate grasp pose");
        stage->properties().configureInitFrom(mtc::Stage::PARENT);
        stage->properties().set("marker_ns", "grasp_pose");
        stage->setPreGraspPose("open");
        stage->setObject(oc.id);//setObject("object");
        stage->setAngleDelta(M_PI / 64); //M_PI is just math.pi
        stage->setMonitoredStage(current_state_ptr);  // Hook into current state

        Eigen::Isometry3d grasp_frame_transform; // affine transformation matrix
        Eigen::Quaterniond q; 
          q = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                             Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
          grasp_frame_transform.linear()= q.matrix();

        grasp_frame_transform.translation().z() = 0.105;    // otherwise the hand will be IN the object
        grasp_frame_transform.translation().x() = 0.0;      
        grasp_frame_transform.translation().y() = 0.0;     
          // Compute IK
        auto wrapper =
            std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
        wrapper->setMaxIKSolutions(2);
        wrapper->setMinSolutionDistance(0.0);
        wrapper->setIKFrame(grasp_frame_transform, hand_frame);
        wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
        wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
        grasp->insert(std::move(wrapper));
      }

      {
        auto stage =
            std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,object)");
        stage->allowCollisions(oc.id,
                              task.getRobotModel()
                                  ->getJointModelGroup(hand_group_name)
                                  ->getLinkModelNamesWithCollisionGeometry(),
                              true);
        grasp->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("close");
        grasp->insert(std::move(stage));
      }


      {
        auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach object");
        stage->attachObject(oc.id, hand_frame);
        attach_object_stage = stage.get();
        grasp->insert(std::move(stage));
      }

      {
        auto stage =
            std::make_unique<mtc::stages::MoveRelative>("lift object", cartesian_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
        stage->setMinMaxDistance(0.00, 0.3);
        stage->setIKFrame(hand_frame);
        stage->properties().set("marker_ns", "lift_object");

        // Set upward direction
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = "world";
        vec.vector.z = 1.0;
        stage->setDirection(vec);
        grasp->insert(std::move(stage));
      }
      
      task.add(std::move(grasp));
    }

    /* place stages*/
    {
      auto stage_move_to_place = std::make_unique<mtc::stages::Connect>(
          "move to place",
          mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner },
                                                    { hand_group_name, sampling_planner } });
      stage_move_to_place->setTimeout(7.0);
      stage_move_to_place->properties().configureInitFrom(mtc::Stage::PARENT);
      task.add(std::move(stage_move_to_place));
    }

    {
      auto place = std::make_unique<mtc::SerialContainer>("place object");
      task.properties().exposeTo(place->properties(), { "eef", "group", "ik_frame" });
      place->properties().configureInitFrom(mtc::Stage::PARENT,
                                            { "eef", "group", "ik_frame" });

                                            {
      // Sample place pose
      auto stage = std::make_unique<mtc::stages::GeneratePlacePose>("generate place pose");
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      stage->properties().set("marker_ns", "place_pose");
      stage->setObject(oc.id);

      geometry_msgs::msg::PoseStamped target_pose_msg;
      target_pose_msg.header.frame_id = "panda_link0";

      geometry_msgs::msg::Pose boxpose;
      boxpose = oc_placebox.primitive_poses.back();

      geometry_msgs::msg::Pose box1_pose;
      geometry_msgs::msg::Pose box2_pose;

      shape_msgs::msg::SolidPrimitive box1_shape;
      shape_msgs::msg::SolidPrimitive box2_shape;

      box1_shape = oc_box.primitives.back();
      box2_shape = oc_placebox.primitives.back();
      box1_pose = oc_box.primitive_poses.back();
      box2_pose = oc_placebox.primitive_poses.back();


      double box1_height;
      double box2_height;

      box1_height = box1_shape.dimensions[2];
      box2_height = box2_shape.dimensions[2];

      geometry_msgs::msg::Pose chicken_pose;
      chicken_pose = oc.mesh_poses.back();
    
      double zmin = box1_pose.position.z + box1_height/2;
      double chicken_bottom_to_center = chicken_pose.position.z - zmin;

      double chicken_center2 = box2_pose.position.z+box2_height/2 + chicken_bottom_to_center;

      target_pose_msg.pose.position.x = boxpose.position.x;  
      target_pose_msg.pose.position.y = boxpose.position.y;
      target_pose_msg.pose.position.z = chicken_center2+0.005;
      
      // set new orientation of chicken
      tf2::Quaternion q_orig, q_rot, q_new;

      q_orig.setRPY(0.0, 0.0, 0.0);
      // choose rotation angle, this is with respect to panda_link0
      q_rot.setRPY(0.0, 0.0, 1.57);   // in our case, we have a pi/2 rotation around the z axis
      q_new = q_rot * q_orig;
      q_new.normalize();
     
      target_pose_msg.pose.orientation.w = q_new[3];
      target_pose_msg.pose.orientation.x = q_new[0];  
      target_pose_msg.pose.orientation.y = q_new[1];
      target_pose_msg.pose.orientation.z = q_new[2];
      
      stage->setPose(target_pose_msg);
      stage->setMonitoredStage(attach_object_stage);  // Hook into attach_object_stage

      // Compute IK
      auto wrapper =
          std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(stage));
      wrapper->setMaxIKSolutions(2);
      wrapper->setMinSolutionDistance(0.0);
      wrapper->setIKFrame(hand_frame);
      wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
      wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
      place->insert(std::move(wrapper));
    } //HIERO

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
      stage->setGroup(hand_group_name);
      stage->setGoal("open");
      place->insert(std::move(stage));
    }

    {
      auto stage =
          std::make_unique<mtc::stages::ModifyPlanningScene>("forbid collision (hand,object)");
      stage->allowCollisions(oc.id,
                            task.getRobotModel()
                                ->getJointModelGroup(hand_group_name)
                                ->getLinkModelNamesWithCollisionGeometry(),
                            false);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach object");
      stage->detachObject(oc.id, hand_frame);
      place->insert(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cartesian_planner);
      stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
      stage->setMinMaxDistance(0.0, 0.3);
      stage->setIKFrame(hand_frame);
      stage->properties().set("marker_ns", "retreat");

      // Set retreat direction
      geometry_msgs::msg::Vector3Stamped vec;
      vec.header.frame_id = "world";
      vec.vector.x = -0.5;
      stage->setDirection(vec);
      place->insert(std::move(stage));
    }

      task.add(std::move(place));
  }

  {
    auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
    stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
    stage->setGoal("ready");
    task.add(std::move(stage));
  }

  return task;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto mtc_task_node = std::make_shared<MTCTaskNode>(options);

  // create executors to keep subscribing to publishers 
  rclcpp::executors::SingleThreadedExecutor executor1;
	auto node1 = std::make_shared<readbox>();   
	executor1.add_node(node1);

  rclcpp::executors::SingleThreadedExecutor executor2;
	auto node2 = std::make_shared<readco>();    // to read the collision object, i.e. the object of interest
	executor2.add_node(node2);

  rclcpp::executors::SingleThreadedExecutor executor3;  // to read the scene, i.e. mesh
	auto node3 = std::make_shared<readscene>();
	executor3.add_node(node3);
  
  rclcpp::executors::SingleThreadedExecutor executor4;
	auto node4 = std::make_shared<readplacebox>();      // to read the placement box
	executor4.add_node(node4);

  // 
  std::vector<std::thread> threads;
  threads.push_back(std::thread([&executor1]() {executor1.spin(); }));
  threads.push_back(std::thread([&executor2]() {executor2.spin(); }));
  threads.push_back(std::thread([&executor3]() {executor3.spin(); }));
  threads.push_back(std::thread([&executor4]() {executor4.spin(); }));
  for (auto &th : threads) {
		th.detach();
	}
  
  using namespace std::this_thread; // sleep_for, sleep_until
  using namespace std::chrono; // nanoseconds, system_clock, seconds


  // WE NEED THIS DELAY BECAUSE OTHERWISE THE OBJECT FRAME CANNOT BE READ AND WE GET AN ERROR
  sleep_for(nanoseconds(10));
  sleep_until(system_clock::now() + seconds(5));


  rclcpp::executors::MultiThreadedExecutor executor;

  auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_task_node]() {
    executor.add_node(mtc_task_node->getNodeBaseInterface());
    executor.spin();
    executor.remove_node(mtc_task_node->getNodeBaseInterface());
  });

  mtc_task_node->setupPlanningScene();
  mtc_task_node->doTask();

  spin_thread->join();
  rclcpp::shutdown();
  return 0;
}
