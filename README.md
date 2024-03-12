# 2024-synthetic-data-binpicking
This is the repository for _Synthetic Data-Based Training of Instance Segmentation: A Robotic
Bin-Picking Pipeline for Chicken Fillets_ by L.M. Jonker, W. Roozing, and N. Strisciuglio.

Each folder contains a separate part of the implementation. The parts are independent.

### franka_control 
Files for implementing the position control.

### generate_synthetic_dataset
Script and Blender files for creating synthetic data.

### mask_rcnn_train_eval
Scripts to train and evaluate Mask R-CNN using our data.

### pipeline
The ROS implementation of the complete bin-picking pipeline
