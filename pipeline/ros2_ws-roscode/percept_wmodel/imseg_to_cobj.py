'''
This script does the following:
1)	Call RealSense camera and capture RGB image as well as depth image.
2)	Perform image segmentation (find chickenfillets) and apply found mask on depth image.
3) 	Get pointcloud of both the full scene as well as the chickenfillets.
4) 	The pointcloud coordinates in step 3) are w.r.t. the camera frame, so get the transform between this frame and 
	the robot base frame panda_link0. Then transform the points to panda_link0.
5) 	We know the location of the chicken w.r.t the base frame, but to be able to grasp the object in MoveIt,
	a CollisionObject message has to be created, which expects the coordinates of the points to be centered around
	the object frame (i.e. chicken frame). So the points are transformed again, now to the chicken frame.
6) 	CollisionObject can be created now. The rotation and translation of the CollisionObject are the transform
	between world and chicken frame.
7) 	Publish pointcloud of scene, object, collision object

To see what it does, launch Rviz first:
>> ros2 launch rs_move launch_rviz.launch.py   # in ws_moveit2/install/rs_move/share/launch/

Then run the node: 
>> ros2 run percept_wmodel test_node

If you really want to see cool stuff, run this after:
>> ros2 launch mtc_tutorial pick_place_demo.launch.py



'''

import rclpy						# allows us to spin nodes
import numpy as np					
from rclpy.node import Node			# can create a node with this
from tf2_ros.buffer import Buffer	# with this the frame transforms can be found
from tf2_ros import BufferInterface
## import files present in dir
from .instance_seg_realsense import *
from .segmentation import load_model
from .add_frame import *

import subprocess
## import message types, s.t. we can publish data
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import shape_msgs.msg as shape_msgs
import geometry_msgs.msg as geometry_msgs
import moveit_msgs.msg as moveit_msgs
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from tf2_ros import TransformException
from tf2_ros import TransformBroadcaster 					# for adding new frame
from tf2_ros.transform_listener import TransformListener 	# for obtaining information on existing frames
from geometry_msgs.msg import Quaternion
import time



def makevar(): # make dictionary including all variables from txt
    dict_vars = {}
    for line in text:                   # read all lines
        if line[0] != "#":
            if "=" in line:             # then make var
                arr = line.split("=")   # after : is the value
                varname = arr[0]
                val = arr[1]
                if "#" in val:          # don't look at comment
                    val = val.replace('\t','')
                val = val.split("#")
                value = val[0].strip()
                dict_vars[varname] = value.replace('\n','')
    return dict_vars


#settingsfile = r"./settings.txt"
with open(settingsfile) as f:
    text = f.readlines()
    set_vars = makevar()

tmodel = set_vars['segmentation_model']
nclasses = int(set_vars['n_classes'])

model = load_model(tmodel,nclasses) 	# load model here, then can be used multiple times

n_poses = int(set_vars['n_views'])  # number of poses the robot will take to capture images


# create node
class PCDPublisher(Node):

	def __init__(self):
		super().__init__('test_node')	
		

		''' create some publishers '''
		# complete pc (pointcloud)
		self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'PC_COMPLETE', 10)

		# object pc
		self.roi_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'PC_ROI', 10)

		# object mesh
		self.mesh_publisher = self.create_publisher(shape_msgs.Mesh, 'ROIMESH',10)

		# object collision object
		self.co_publisher = self.create_publisher(moveit_msgs.CollisionObject, 'CO',10)

		self.co_scene_publisher = self.create_publisher(moveit_msgs.CollisionObject, 'CO_SCENE',10)
		# object collision object box
		self.co_box_publisher = self.create_publisher(moveit_msgs.CollisionObject, 'CO_BOX',10)
		# object collision object PLACEbox
		self.co_placebox_publisher = self.create_publisher(moveit_msgs.CollisionObject, 'CO_PLACEBOX',10)


		# publisher for chicken frame
		self.tf_broadcaster = TransformBroadcaster(self)	
		timer_period = 1  # seconds
		self.timer = self.create_timer(timer_period, self.timer_callback)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		# self.timer = self.create_timer(1.0,self.timer_callback)

		# get transform w.r.t. world frame
		# 'target_frame' is the base frame, in this case panda_link0 (declared later), 'cam_frame' is camera frame

		# self.target_frame = self.declare_parameter('target_frame', 'panda_hand').get_parameter_value().string_value
		self.target_frame = self.declare_parameter('target_frame', 'cam_frame').get_parameter_value().string_value
		self.get_new_pc = False
		self.get_camdata = True
		# self.posecounter = 0
		
		self.process = True

	def timer_callback(self):
		
		loop_for = range(n_poses)
		if self.process:
			for posecounter in loop_for:
				new_not_points = []
				if posecounter > 0:
					subprocess.run(["./src/percept_wmodel/percept_wmodel/move_cam.sh", str(posecounter-1)]) # maybe jsut feed the number to this command
					# i.e. only move robot after picture is taken in starting position

				'''
				timer_callback is called every timer_period.
				Here, the frame transforms are obtained, and the points are converted to the right frame (base, i.e. panda_link0).
				'''
				from_frame_rel = self.target_frame	# target_frame is cam_frame
				to_frame_rel = 'panda_link0'		# panda_link0 is base frame

				
				try:
					'''
					Here we get the transform between the camera frame and base frame. We have all points expressed in 
					coordinates which are w.r.t. the camera frame, then with the transform we can convert the points s.t. they
					are expressed in the base frame, which is needed for the control part.
					'''
					
					tr = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel,rclpy.time.Time())
					tr_t = True


				except TransformException as ex:
					self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
					return

				proj_frame = tr.header.frame_id 

				geom = tr.transform 										# geometry_msgs/Transform
				transl_transform = get_loc_from_transf(geom.translation)  	# location of cam frame in panda_link0 coords
				q_transform = get_q_from_transf(geom.rotation) 				# quaternion

				if self.get_camdata:
					roi, pts, frame_rot, frame_trans, vertices, triangles = key_in(model) 	# obtain full + object point cloud + object frame and mesh
					
					# add 'self.': can use the points etc. in timer_callback now
					self.points = pts					# full point cloud
					self.roi = roi						# point cloud of chicken
					self.frame_rot = frame_rot			# rotation of chicken frame w.r.t. camera frame
					self.frame_trans = frame_trans		# transformation of chicken frame w.r.t. camera frame
					
					self.get_new_pc = True		# is true if we do not have a filtered point cloud yet
					self.get_camdata = False

					# get pointcloud in world coordinates
					new_pc = get_new_pc(self.points, q_transform, transl_transform)	# full point cloud
					new_roi = get_new_pc(self.roi, q_transform, transl_transform)	# object point cloud


				if self.get_new_pc:
					'''
					If the robot moves, we do not want the scene to move as well. However, when the camera frame moves, 
					the points will automatically move with it. So we only compute the pc + collision object of the chicken
					once. The coordinates will not be updated when the robot moves. 
					'''
					self.new_roi2,self.vertices,self.triangles = filter_pc(new_roi,False,0.001,True)	# get filtered pc + mesh 
					
					
					self.get_new_pc = False														# s.t. we won't do this again

					if posecounter == 0:
						self.multi_view_pc = new_pc
						self.multi_view_object = self.new_roi2
					else:
						self.multi_view_pc = np.concatenate((self.multi_view_pc,new_pc),axis=0)
						self.multi_view_object = np.concatenate((self.multi_view_object,self.new_roi2),axis=0)

					self.pcd = point_cloud(self.multi_view_pc, 'panda_link0') 			# ros message of point cloud# will show the same as point_cloud(self.points,'cam_frame')
					self.pcd_roi = point_cloud(self.multi_view_object, 'panda_link0')  	# will show the same as point_cloud(self.roi, 'cam_frame')
				
					self.pcd_publisher.publish(self.pcd)		# complete pc publisher
					self.roi_publisher.publish(self.pcd_roi)  	# object pc publisher


				# define new object frame
				t = TransformStamped()  ## of current robot position
				t.header.stamp = self.get_clock().now().to_msg()
				t.header.frame_id = 'panda_link0'					# the new frame is created in panda_link0
				t.child_frame_id = 'chicken_frame'  				# name of the new frame		
			
				# calculate chicken frame rotation + translation w.r.t. panda_link0
				obj_rot, self.obj_loc = pcframe(self.multi_view_object)#.new_roi2) # obtain the object frame (rotation + translation)

				# get right quaternion based on the rotation angles
				q_orig = quaternion_from_euler(0, 0, 0)
				self.q_rot = quaternion_from_euler(obj_rot[0],obj_rot[1],obj_rot[2])	
				self.q_new = quaternion_multiply(self.q_rot, q_orig)

				# t is in world frame coordinates
				t.transform.translation.x = self.obj_loc[0] 
				t.transform.translation.y = self.obj_loc[1]
				t.transform.translation.z = self.obj_loc[2]
				t.transform.rotation.w = self.q_new[0]
				t.transform.rotation.x = self.q_new[1] 
				t.transform.rotation.y = self.q_new[2]
				t.transform.rotation.z = self.q_new[3]   	


                '''
				moveit_msgs.CollisionObject expects an object with its center at (0,0,0) and a rotation + translation
				w.r.t. its base frame. So the points need to be converted to chicken_frame. For this, an inverse quaternion 
				multiplication is done, as we have the transformation from world to chicken_frame. 
				'''
				self.new_roi_ch_frame = inverse_transf(self.multi_view_object,self.q_new,self.obj_loc)
				self.new_roi3,self.vertices3,self.triangles3 = filter_pc(self.new_roi_ch_frame,False,0.001,True)	# get mesh
				self.tf_t = t


				self.new_scene = inverse_transf(self.multi_view_pc, self.q_new,self.obj_loc)
				self.scene_vertices, self.scene_triangles = surface_rec(self.new_scene)

				if posecounter == len(loop_for)-1:
					self.get_camdata = False
					self.process = False
				else:
					self.get_camdata = True



		self.tf_t.header.stamp = self.get_clock().now().to_msg()
		self.tf_broadcaster.sendTransform(self.tf_t)


		# publish both the complete point cloud, as well as the object point cloud, in INERTIAL FRAME
		self.pcd_publisher.publish(self.pcd)		# complete pc publisher
		self.roi_publisher.publish(self.pcd_roi)  	# object pc publisher

		# create mesh and collision objects, IN OBJECT FRAME
		self.mesh, self.co, self.obj_hullc = create_mesh(self.vertices3, self.triangles3, self.q_new,self.obj_loc,1)
	
		self.box = create_boxmesh(self.multi_view_object,self.obj_loc,[0.5,0.8,0.03],False)

		# create mesh of rest of environment
		self.not_mesh, self.not_co, self.not_hullc = create_mesh(self.scene_vertices,self.scene_triangles, self.q_new,self.obj_loc,2)#self.not_vertices_ch, self.not_triangles_ch, self.not_roi_ch, self.q_new,self.obj_loc)


		place_loc = [0.6,0.0,0.05]
		self.placebox = create_boxmesh(self.multi_view_object, place_loc,[0.1,0.2,0.01], False)
		# publish mesh and collision objects
		# self.mesh_publisher.publish(self.mesh) 
		self.co_scene_publisher.publish(self.not_co)
		self.co_box_publisher.publish(self.box)#
		self.co_placebox_publisher.publish(self.placebox)
		self.co_publisher.publish(self.co)
		self.offset_publisher.publish(self.obj_hullc)


	
	


def create_boxmesh(points,translation,dims,dir_below):
	'''
	From points in world coordinates, get max and min z-values.
	Then using translation in x and y, create a boxmesh which should
	represent a table on which the chicken fillets are positioned.
	'''
	co_box = moveit_msgs.CollisionObject()
	co_box.header.frame_id = "panda_link0"
	co_box.id = "box1"
	

	box = shape_msgs.SolidPrimitive()
	box.type = 1 # box
	box.dimensions = dims

	zmax = np.amax(points[:,2]) # get max z value
	zmin = np.amin(points[:,2])	# get min z value

	pose = geometry_msgs.Pose()
	pose.orientation.w = 1.0
	pose.position.x=translation[0]
	pose.position.y=translation[1]

	if dir_below:
		pose.position.z= zmin - (box.dimensions[2]/2)
	else:
		pose.position.z = translation[2]
	co_box.primitives.append(box)
	co_box.primitive_poses.append(pose)
	co_box.operation = co_box.ADD
	return co_box

def get_q_from_transf(rotation):
	''' The rotation of the transformation is quaternion type, which has x,y,z entries. 
	In the rest of the qode the quaternion is indexed [], so transform it to this.'''
	q = np.empty(4)
	q[0] = rotation.w
	q[1] = rotation.x
	q[2] = rotation.y
	q[3] = rotation.z
	
	return q

def get_loc_from_transf(translation):
	''' The translation from the transformation contains x,y,z entries, make this into an array.'''
	loc = np.zeros(3)
	loc[0] = translation.x
	loc[1] = translation.y
	loc[2] = translation.z

	return loc


def inverse_transf(points,rotation,location):
	num_rows, num_cols = points.shape
	new_points = np.concatenate((np.array(np.zeros(len(points)))[:, np.newaxis], points), axis=1)  #[0,x,y,z]	
	
	q = rotation
	qinv = np.array([q[0],-q[1],-q[2],-q[3]])

		#print('newpoints without extra tr:', new_points)
	new_points[:,1]-=location[0]
	new_points[:,2]-=location[1]
	new_points[:,3]-=location[2]

	for i in range(len(new_points)):
		new_points[i,:] = quaternion_multiply(quaternion_multiply(qinv,new_points[i,:]),q)
	
	new_points = np.delete(new_points, 0, axis = 1)
	
	return new_points
	
	
def get_new_pc(points,rotation,translation):
	''' Given a set of points (Nx3) and a rotation+translation w.r.t. a different frame, 
	calculate the new values of the points given the transformation.

	OUT:
	The rotation should be a quaternion.
	The translation a 1x3 array.
	
	newpoints = q*P*qinv + translation'''
	num_rows, num_cols = points.shape
	new_points = np.concatenate((np.array(np.zeros(len(points)))[:, np.newaxis], points), axis=1)  #[0,x,y,z]	

	q = rotation

	qinv = np.array([q[0],-q[1],-q[2],-q[3]])

	for i in range(len(new_points)):
		new_points[i,:] = quaternion_multiply(quaternion_multiply(qinv,new_points[i,:]),q)

	new_points[:,1]+=translation[0]
	new_points[:,2]+=translation[1]
	new_points[:,3]+=translation[2]

	new_points = np.delete(new_points, 0, axis = 1)

	return new_points # first column is zeros anyway

	
def multiarray(arr):
	'''
	Create multiarray object
	'''
	multiarr = std_msgs.Float32MultiArray(data = arr)
	return multiarr

def create_mesh(verts,tris,rotation,center,counter): # inputs are both the convex hull, as well as the pointcloud of the object
	'''	
	Create mesh and collision object messages, given vertices and triangles.

	'''
	mesh = shape_msgs.Mesh()

	skip = []

	for i in range(len(verts)):
		v =  geometry_msgs.Point()
		v.x= float(verts[i,0]) # have to add float() because type is now np.float, vertices only accepts float
		v.y= float(verts[i,1])
		v.z= float(verts[i,2])
		mesh.vertices.append(v)


	## this is how much the center of the chicken differs from its origin
	cx = np.mean(verts[:,0])
	cy = np.mean(verts[:,1])
	cz = np.mean(verts[:,2])
	hc = [cx,cy,cz]
	
	hullcentre = multiarray(hc)
	for i in range(len(tris)):
		t = shape_msgs.MeshTriangle()
		#print(type(ch[i,:]))
		t.vertex_indices = np.array(tris[i,:], dtype = np.uint32)
		mesh.triangles.append(t)

	co = moveit_msgs.CollisionObject()
	co.header.frame_id = "panda_link0"
	co.id = "obj" + str(counter)
	co.meshes.append(mesh)

	# translation and rotation are 0, because the data is already wrt the cam_frame

	mposes = geometry_msgs.Pose()
	mposes.orientation.w = rotation[0]  ## are these wrt world frame?
	mposes.orientation.x = rotation[1]
	mposes.orientation.y = rotation[2]
	mposes.orientation.z = rotation[3]

	mposes.position.x=center[0]
	mposes.position.y=center[1]
	mposes.position.z=center[2]
	co.mesh_poses.append(mposes)
	co.operation = co.ADD
	
	
	return mesh, co, hullcentre


def point_cloud(points, parent_frame):  ## https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
	''' Convert a point cloud (Nx3 array) into a point cloud message, which can be subscribed to in e.g. RViz.
	The parent_frame should equal the frame in which the points are expressed (camera frame, base frame?).
	'''


	ros_dtype = sensor_msgs.PointField.FLOAT32
	dtype = np.float32
	itemsize = np.dtype(dtype).itemsize
	data = points.astype(dtype).tobytes()
	fields = [sensor_msgs.PointField(name = n, offset = i * itemsize, datatype = ros_dtype, count = 1) for i, n in enumerate('xyz')]
	header = std_msgs.Header(frame_id = parent_frame)
	
	return sensor_msgs.PointCloud2(
		header = header,
		height = 1,
		width = points.shape[0],
		is_dense = False,
		is_bigendian = False,
		fields = fields,
		point_step = (itemsize * 3),
		row_step = (itemsize * 3* points.shape[0]),
		data = data
		)

def main(args=None):
	rclpy.init(args=args)
	pcd_publisher = PCDPublisher()
	rclpy.spin(pcd_publisher)
	rclpy.shutdown()
	
if __name__ == '__main__':
	main()	
