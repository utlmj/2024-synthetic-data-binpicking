from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node
import numpy as np

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import Quaternion
import math


## https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
def quaternion_from_euler(ar, ap, ay):
	ar /= 2.0
	ap /= 2.0
	ay /= 2.0
	cr = math.cos(ar)
	sr = math.sin(ar)
	cp = math.cos(ap)
	sp = math.sin(ap)
	cy = math.cos(ay)
	sy = math.sin(ay)
	#cc = cr*cy
	#cs = cr*sy
	#sc = sr*cy
	#ss = sr*sy

	q = np.empty((4, ))
	q[0] = cr*cp*cy + sr*sp*sy#cp*cc + sp*ss
	q[1] = sr*cp*cy - cr*sp*sy#cp*sc - sp*cs
	q[2] = cr*sp*cy + sr*cp*sy#cp*ss + sp*cc
	q[3] = cr*cp*sy - sr*sp*cy#cp*cs - sp*sc
	# THIS WORKS
	return q # w,x,y,z
    
def quaternion_multiply(q0, q1):
    """
    Multiplies two quaternions.

    Input
    :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
    :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from q0
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]

    # Extract the values from q1
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]

    # Computer the product of the two quaternions, term by term
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


class FixedFrameBroadcaster(Node):

	def __init__(self):
		super().__init__('add_frame')
		self.tf_broadcaster = TransformBroadcaster(self)
		self.timer = self.create_timer(0.1, self.broadcast_timer_callback)

	def broadcast_timer_callback(self):
		t = TransformStamped()

		t.header.stamp = self.get_clock().now().to_msg()
		t.header.frame_id = 'panda_hand'
		t.child_frame_id = 'cam_frame'
		## in rviz : X axis is indicated in red, the Y axis is indicated in green, and the Z axis is indicated in blue.

		q_orig = quaternion_from_euler(0, 0, 0)
		
		q_rot = quaternion_from_euler(0,  0, 1.5708)  # units are radians; this implies a rotation of pi/2 around the z-axis
		
		
		q_new = quaternion_multiply(q_rot, q_orig)
		# print(q_new)

		t.transform.translation.x = 0.065
		t.transform.translation.y = -0.045
		t.transform.translation.z = 0.0292
		t.transform.rotation.w = q_new[0]
		t.transform.rotation.x = q_new[1]
		t.transform.rotation.y = q_new[2]
		t.transform.rotation.z = q_new[3]   
		


		self.tf_broadcaster.sendTransform(t)
		print(t)

def main():
    rclpy.init()
    node = FixedFrameBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
