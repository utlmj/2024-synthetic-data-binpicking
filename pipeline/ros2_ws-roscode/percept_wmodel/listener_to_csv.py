'''
Run this node in a separate terminal. It listens to the joint states of the panda robot in Rviz.
Then it saves the data in a csv.
'''

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

import re
import csv

from ast import literal_eval
import pandas as pd
from pathlib import Path
import os
import sys


class MinimalSubscriber(Node):
    

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',#'/panda_arm_controller/joint_trajectory',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning


        ## init dataframe to save positions
        self.data = ['time_sec','time_nanosec',
            'panda_joint1',
            'panda_joint2',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
            'panda_finger_joint1',
            'panda_joint3',
            'panda_finger_joint2']
        self.df = pd.DataFrame(columns = self.data)
        self.count = 0

        filenum = 0
        dir = "/home/csv/oldcsv/"
        filename = "joint_pos_path_"
        filetype = ".csv"
      
        while True:
            fullpath = dir + filename + str(filenum)+filetype

            path_exists = os.path.exists(fullpath)

            if path_exists: # because we do not want to overwrite an existing file
                filenum += 1
            else:
                self.filepath = Path(fullpath)  
                break
        
        # we only use successpath if the execution was successful
        self.successpath = Path(dir+"success/"+filename+str(filenum)+filetype)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)  
        self.df.to_csv(self.filepath)  

    def listener_callback(self, msg):

        try:
            self.get_logger().info('I heard something!')

            ## add data to dataframe
            time_arr = re.findall(r'\d+', str(msg.header.stamp))
            time_sec = str(time_arr[0])
            time_nanosec = str(time_arr[1])
            print('TIMESec',time_sec) 
            print('TIMENANOSEC', time_nanosec)
            self.df.loc[self.count] = [time_sec,time_nanosec,
                str(msg.position[0]),
                str(msg.position[1]),
                str(msg.position[2]),
                str(msg.position[3]),
                str(msg.position[4]),
                str(msg.position[5]),
                str(msg.position[6]),
                str(msg.position[7]),
                str(msg.position[8])]
            self.count += 1
            print(self.df)
            self.df.to_csv(self.filepath,index=False) # save dataframe as csv file


        except KeyboardInterrupt:
            print('Interrupted')
            success = input("Execution successfull? y/n ")
            if success == "y":
                self.successpath.parent.mkdir(parents=True, exist_ok=True)  
                self.df.to_csv(self.successpath,index=False)  
                print("File saved in ", self.successpath)
            self.df.to_csv(self.filepath,index=False) # save dataframe as csv file
            print("File saved in ", self.filepath)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)

    # destroy node
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
