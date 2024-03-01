###############################################################
### PERFORM INSTANCE SEGMENTATION ON REALSENSE DATA ###
###############################################################

'''
Get data from RealSense camera + process data.
Instance segmentation on RGB image, then apply mask to depth image --> segmented pointcloud.

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
#import keyboard
import time
import statistics
from .segmentation import *
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy import signal
import pyrealsense2.pyrealsense2 as rs
#import pyvista as pv
import rclpy

import open3d as o3d

from .add_frame import *

from datetime import datetime # to save images


plt.close('all')


def key_in(model): 
	while(True):

		masks = []
		while masks == []: # so if we do not see an object in the images, do again
			rgb, depth , dframe, rframe = get_img()
			masks, rect, score = do_segmentation(rgb,model)
		

		now = datetime.now()
		saveimas = "scene_" + str(now) + ".jpg"
		# cv2.imwrite(saveimas,rgb)

		selected_instance_idx = select_instance(masks, score, depth) # output will be one of the input masks
		m = masks[selected_instance_idx] > 0.5 #m = masks[0] > 0.5

		kernel = np.ones((5, 5), np.uint8)
		me = np.array(m,dtype=np.uint8)
		m = cv2.erode(me,kernel)
		
		vertices, texcoords = get_pointcloud(dframe, rframe)#get_pointcloud(dframe,rframe)

		testvar = 5
		
		roi,ind = get_roi(m,vertices,True)
		rest,idx = get_roi(m,vertices,False) # get roi of the rest, i.e. everything BUT the mask
			
		### filter point cloud (remove outliers etc.) + get convex hull vertices and triangles
		filtered_pc, hullvertices, triangles = filter_pc(roi,False,0.003,False)		
		frame_rot, frame_trans = pcframe(filtered_pc)#roi) # get rotation+translation of chicken in cam frame
		
		## 2nd line is deleting a bit more than the selected chicken mask
		new_vertices = np.delete(vertices,ind[0],axis=0) # pointcloud without chicken no?			

		return roi, rest, frame_rot, frame_trans, hullvertices, triangles#, scene_vertices, scene_triangles#, not_selected_chicken_points
		# roi = point cloud of selected chicken
		# new_vertices = point cloud of rest of scene, except selected chicken
		# frame_rot = frame rotation  of selected chicken, w.r.t. camera frame
		# frame_trans = frame translation of selected chicken, w.r.t. camera frame
		# hullvertices = vertices of convex hull of selected chicken
		# triangles = triangles of convex hull of selected chicken
		# not_selected_chicken_vertices = list of vertices of convex hull of not selected chicken
		# not_selected_chicken_triangles = list of triangles of convex hull of not selected chicken
		# not_selected_chicken_points = list of points of not selected chicken


def surface_rec(points):
	'''
	reconstructs the surface of pointcloud
	Not the same as convex hull, here we try to not fill holes


	'''
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
	# o3d.visualization.draw_geometries([voxel_down_pcd])
	tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(voxel_down_pcd)
	# for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
	alpha = 0.01
	print(f"alpha={alpha:.3f}")
	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
		voxel_down_pcd, alpha, tetra_mesh, pt_map)
	mesh.compute_vertex_normals()
	o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)

	vertices = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.triangles)

	print(vertices)

	return vertices,triangles
		
def get_roi(m,vertices,selected_mask):
	''' 
	inputs
	m: mask of selected chicken
	vertices: full scene point cloud
	selected_mask: True or False, True if we want the ROI of the selected chicken
		False if we want the rest

	returns
	roi: points where mask m OR img_erosion are True, depending on value of selected_mask
	ind: indices where either the chicken is, or the rest, depending on value of selected_mask
	'''
	if selected_mask:
		ind = np.where(m.reshape(vertices.shape[0]))
	else:
		testmask = (np.array(m,dtype=bool))

		## apply erosion
		kernel2 = np.ones((5, 5), np.uint8)
		m = (np.array(np.invert(testmask)*1,dtype=bool)*1)/255

		img_erosion = cv2.erode(m, kernel2, iterations=2)
		# plt.figure(), plt.imshow(img_erosion), plt.show()
		ind = np.where(img_erosion.reshape(vertices.shape[0])) ## just select everything else

	roi = vertices[ind[0],:]
	# plt.imshow(m)
	# plt.show()

	return roi, ind

def select_instance(masks, score, depth):
	'''
	inputs
	masks: all predices masks (list)
	score: all scores per mask (list)
	depth: depth image from realsense

	returns
	idx: indicates which mask in masks is selected (masks[idx])

	'''
	check_overlap = True
	min_depth = []
	mid = []
	grasp_score = []
	save_mask = []
	n = 0
	for i in range(len(masks)):
		n+=1
		mask = masks[i] > 0.5
		# plt.imshow(mask)
		obj = depth.copy()
		temp = depth.copy()
		obj[:,:,0] = temp[:,:,0]*mask
		obj[:,:,1] = temp[:,:,1]*mask
		obj[:,:,2] = temp[:,:,2]*mask

		# obj is masked depth map, only values at where there is a chicken fillet
		print(obj[obj>0])
		min_depth.append(np.mean(obj[obj > 0])) # or should we look at mean value? min value is 'least' depth, i.e. closest by
		# plt.figure()
		# plt.imshow(obj)
		# plt.show()

		ind = np.where(obj)
		x = ind[1]
		y = ind[0]
		centerx = np.mean(x)
		centery = np.mean(y)
		center = [centerx,centery]
		cx = centerx-(np.shape(obj)[1]/2)
		cy = centery-(np.shape(obj)[0]/2)

		mid.append((abs(cx) + abs(cy)))

		# based on the depth, area, and score, calculate 'most graspable instance'
		w1 = 0.0	# score weight
		w2 = 0.9	# depth weight
		w3 = 0.1
		
	p = mid[0]
	for k in range(len(masks)-1):
		j = abs(p - mid[k+1]) 

		if j < 10:
			w3 = 0.0
			

	for k in range(len(masks)):
		grasp_score.append(w1*score[k] + w2*(1/min_depth[k]) + w3*(1/mid[k]))
		# add weights to different components?
		print(grasp_score[k])

	# plt.show()

	idx = np.argmax(grasp_score)
	return idx
	


def filter_pc(cloud, cluster,size,do_plot):
	'''
	inputs
	cloud: pointcloud of interest
	cluster: bool, True if we want to cluster the pointcloud
	size: voxel size
	do_plot: bool, True if we want to see the plots

	returns
	out_cloud: filtered (+ clustered) pointcloud 
	hullvertices: vertices of convex hull of pc
	triangles: triangles of convex hull of pc
	http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html?highlight=convex%20hull#Convex-hull
	'''
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(cloud)
	voxel_down_pcd = pcd.voxel_down_sample(voxel_size=size)
	cropped,ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=40,std_ratio=3.0) 
	#http://www.open3d.org/docs/0.8.0/tutorial/Advanced/pointcloud_outlier_removal.html
	### create meTruesh
	if do_plot:
		o3d.visualization.draw_geometries([voxel_down_pcd])
	inlier_cloud = voxel_down_pcd.select_by_index(ind)
	if do_plot:
		o3d.visualization.draw_geometries([inlier_cloud])


	pcd = inlier_cloud#.sample_points_poisson_disk(5000)
	pcd.normals = o3d.utility.Vector3dVector(np.zeros(
		(1, 3)))  # invalidate existing normals
	# http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Normal-Estimation
	pcd.estimate_normals()
	#o3d.visualization.draw_geometries([pcd], point_show_normal=True) # show normals



	if cluster:
		# edit this part
		labels = np.array(pcd.cluster_dbscan(eps=0.07, min_points = 10, print_progress=False))
		print('labels',labels)
		max_label = labels.max()
		print("%d clusters" % (max_label + 1))
		idx = np.where(labels == 0)
		print('INDEX',idx)
		pcd_clustered = pcd.select_by_index(idx[0])
	else: 
		pcd_clustered = pcd

	
	hull, _ = pcd_clustered.compute_convex_hull()

	# only to plot ch
	hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
	hull_ls.paint_uniform_color((1, 0, 0))
	if do_plot:
		o3d.visualization.draw_geometries([pcd_clustered, hull_ls])
	##
	hullvertices = np.asarray(hull.vertices)
	triangles = np.asarray(hull.triangles)

	# extract vertices and points
	out_cloud = np.asarray(inlier_cloud.points)
	# print(np.asarray(inlier_cloud.points))
	return out_cloud, hullvertices, triangles


def get_img():
	'''
	get image data from RealSense

	returns
	color_image: 
	depth_colormap: 
	depth_frame:
	color_frame:
	'''
	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()

	# Get device product line for setting a supporting resolution
	pipeline_wrapper = rs.pipeline_wrapper(pipeline)
	pipeline_profile = config.resolve(pipeline_wrapper)
	device = pipeline_profile.get_device()
	device_product_line = str(device.get_info(rs.camera_info.product_line))

	found_rgb = False
	for s in device.sensors:
		if s.get_info(rs.camera_info.name) == 'RGB Camera':
			found_rgb = True
			break
	if not found_rgb:
		print("The demo requires Depth camera with Color sensor")
		exit(0)



	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

	if device_product_line == 'L500':
		config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
	else:
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	# Start streaming
	profile = pipeline.start(config)


	# Getting the depth sensor's depth scale (see rs-align example for explanation)
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	print("Depth Scale is: " , depth_scale)

	# We will be removing the background of objects more than
	#  clipping_distance_in_meters meters away
	clipping_distance_in_meters = 10 #1 meter
	clipping_distance = clipping_distance_in_meters / depth_scale


	align_to = rs.stream.color
	align = rs.align(align_to)


	# start timer
	start = time.time()
	while True:        

		frames = pipeline.wait_for_frames()
		aligned_frames = align.process(frames)
		color_frame = aligned_frames.get_color_frame()
		depth_frame = aligned_frames.get_depth_frame()

		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())#*depth_scale
		print('Average depth: ',np.mean(depth_image*depth_scale))
		color_image = np.asanyarray(color_frame.get_data())


		# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_RAINBOW)

		depth_colormap_dim = depth_colormap.shape
		color_colormap_dim = color_image.shape


		# If depth and color resolutions are different, resize color image to match depth image for display
		if depth_colormap_dim != color_colormap_dim:
			resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
			images = np.hstack((resized_color_image, depth_colormap))
		else:
			images = np.hstack((color_image, depth_colormap))

		# Show images
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', images)
		cv2.waitKey(1)

		end = time.time()
		if end-start > 5:
			cv2.destroyAllWindows()
			break


	# implement: wait until robot arm is at desired location, then exit try


	return color_image, depth_colormap, depth_frame, color_frame#, verts, texcoords

def get_pointcloud(depth_frame,rgb_frame):
	pc = rs.pointcloud()
	pc.map_to(rgb_frame)
	points = pc.calculate(depth_frame)
	# Pointcloud data to arrays
	v, t = points.get_vertices(), points.get_texture_coordinates()
	verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # vertices of point cloud (x,y,z)
	texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv map
	#print(texcoords.shape)
	#print(verts.shape)
	return verts, texcoords
    


def var(arr):
	'''
	computes the variance of an array
	'''
	mean = np.mean(arr)
	diff = arr - mean
	variance = np.sum(np.square(diff))/len(arr)
	print('VARIANCE: ', variance)
	return variance
 
def pca(pointclouddata):
	'''
	performs pca on 3D pointcloud
	'''

	centerx, centery, centerz = np.mean(pointclouddata, axis = 0)
	centerz = np.median(pointclouddata[:,2])
	center = np.array([centerx,centery,centerz]) # centerpoint of pointcloud
	print(pointclouddata.shape)
	varx = var(pointclouddata[:,0])
	vary = var(pointclouddata[:,1])
	varz = var(pointclouddata[:,2])
	print('MAX',np.amax(pointclouddata[:,2]))
	print('MIN',np.amin(pointclouddata[:,2]))

	n = len(pointclouddata[:,0])  # get number of points
	zmean = pointclouddata - center  # zero mean data
	print(zmean.shape)
	print(pointclouddata)
	print(zmean)
	C = (1/(n-1)) * np.dot(zmean.transpose(),zmean) 

	eigval, eigvec = np.linalg.eig(C) # the eigenvectors of C are the directions of the principal components, the magnitude of the eigenvalues determine which is the 1st, 2nd, 3rd pc
	return center, eigval, eigvec 
    

def pcframe(pointclouddata):
	'''
	after using pca, the eigenvector can be converted to euler angles --> frame
	'''
	#pointclouddata = pointclouddata[::100,:]
	center, eigval, eigvec = pca(pointclouddata) #### this is wrt camframe
	print('CENTER',center)
	print(eigval)
	print(eigvec)

	ind = np.argsort(-eigval) # argsort sorts starting with the smallest number, so will sort by largest number if we negate the array
	print('index',ind)
	eigval = np.sort(eigval)
	eigval = eigval[::-1]
	print('SORTED',eigval)
	ind3 = np.tile(ind, (3,1))
	print('newind',ind3)
	eigvec = np.take_along_axis(eigvec, ind3, axis = 1)
	print('SORTED',eigvec)

	# the eigenvectors represent the direction of the axes, starting at the centerpoint


	rot_euler = rotationMatrixToEulerAngles(eigvec)
	print(' EULER' , rot_euler)
	#rot_euler = [x for _,x in sorted(zip(ind,rot_euler))]
	#print(' EULER_sorted' , rot_euler)
	transl = center.astype(float)
	return rot_euler, transl

    

##### below is from https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
    
    
def rotationMatrixToEulerAngles(R) :
	assert(isRotationMatrix(R))
	sy = math.sqrt(R[2,1] * R[2,1] +  R[2,2] * R[2,2])
	singular = sy < 1e-6
	if  not singular:
		x = -math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = -math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	return np.array([x, y, z])