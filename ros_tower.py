import rospy
import random
import pickle
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
# import tf2_ros
# import geometry_msgs.msg
import os
import paramiko

import pygame
from gtts import gTTS
import io
import threading
import ast

experiment_number = 0
experiment_subsection = 0


class PPSLogger():
    def __init__(self, sub_type="Single"):
        self.tactile_pub = rospy.Publisher('/tower_command', String, queue_size=10)
        self.moveit_pub = rospy.Publisher("/graspflow/move_to", String, queue_size=1)
        self.ssh = paramiko.SSHClient() 
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # self.ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        self.ssh.connect('10.245.125.242', username='clear', password='crslab2017!')

    def ros_tower(self):
        print("Initializing ROS tower")
        while not rospy.is_shutdown():
            query = input("Query: ")
            if query == "exit":
                break
            elif query == 'capture':
                data = []
                
                # for i in range(1, 4):
                # for i in range(5):
                #     msg = String()
                #     msg.data = str("capture {}".format(i))
                #     self.tactile_pub.publish(msg)
                    
                #     rospy.sleep(3)
                #     rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
                #     aligned_depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)

                #     data.append(rgb_msg.data, aligned_depth_msg.data)
                #     print('Done with view {}.'.format(i))

                # np.save("pc_everything.npy", np.array(data))
                msg = String()
                msg.data = str("capture")
                self.tactile_pub.publish(msg)
                rospy.sleep(5)
                ### move your top down image here
                sftp = self.ssh.open_sftp()
                sftp.get('/home/clear/catkin_ws/images/top_down_img.png', '/home/crslab/Grounded-Segment-Anything/images/top_down_img.png')
                print("Copied top down image")
                rospy.sleep(20)
                ### TODO SSH HERE
                source_folder="/home/clear/catkin_ws/images/"
                inbound_files=sftp.listdir(source_folder)
                for file in inbound_files :
                    filepath = source_folder+file
                    localpath = "/home/crslab/Grounded-Segment-Anything/images/" + file
                    sftp.get(filepath, localpath)
                sftp.get('/home/clear/catkin_ws/rgb.npy', '/home/crslab/Grounded-Segment-Anything/rgb.npy')
                sftp.get('/home/clear/catkin_ws/depth.npy', '/home/crslab/Grounded-Segment-Anything/depth.npy')
                
                sftp.close()

            elif query == 'scan':
                msg = String()
                msg.data = str("0 0 scan")
                self.moveit_pub.publish(msg)
                rospy.sleep(5)
                ### move your top down image here
                sftp = self.ssh.open_sftp()
                sftp.get('/home/clear/catkin_ws/images/top_down_img.png', '/home/crslab/Grounded-Segment-Anything/images/top_down_img.png')
                sftp.get('/home/clear/catkin_ws/images/depth.png', '/home/crslab/Grounded-Segment-Anything/images/top_down_depth_img.png')
                print("Copied top down image")

                sftp.get('/home/clear/catkin_ws/rgb.npy', '/home/crslab/Grounded-Segment-Anything/rgb.npy')
                sftp.get('/home/clear/catkin_ws/depth.npy', '/home/crslab/Grounded-Segment-Anything/depth.npy')
                
                sftp.close()
                print("Copied all needed files")

            ### ready microphone
            elif 'grasp' in query:
                ##  move filter here
                all_t = np.load("/home/crslab/cehao/data/grasp/tran.npy")
                all_r = np.load("/home/crslab/cehao/data/grasp/rot.npy")

                index = 0

                if len(query.split(" ")) > 1:
                    index = int(query.split(" ")[1])

                file_path = '/home/crslab/cehao/data/prompt.txt'
                file_path = '/home/crslab/kelvin/GraspGen/item_prompt.txt'
                with open(file_path, 'r') as f:
                    obj_answer = ast.literal_eval(f.read())
                    target = obj_answer['target']
                    is_in_grasp = obj_answer['grasp']

                    if target == 'handover':
                        target = 0
                    elif target == None:
                        target = 1
                    elif target == 'table':
                        target = 3
                    else:
                        target = 2
                
                target_trans = [0, 0, 0]
                target_rot = [0.999, -0.046, 0.018, -0.008]

                if target == 2:
                    target_trans = np.load("/home/crslab/cehao/data/grasp/secondary_tran.npy")

                
                should_execute = 4 if is_in_grasp else 3

                target_msg_use = '{} {} [[{},{},{},{},{},{},{}], [{},{},{}], [{},{},{},{}]]'.format(should_execute,
                                                                                                    target,target_trans[0],
                                                                                                   target_trans[1],
                                                                                                    target_trans[2],
                                                                                                    target_rot[0],
                                                                                                    target_rot[1],
                                                                                                    target_rot[2],
                                                                                                    target_rot[3],
                                                                                                   all_t[index][0],
                                                                                        all_t[index][1],
                                                                                        all_t[index][2],
                                                                                        all_r[index][0],
                                                                                        all_r[index][1],
                                                                                        all_r[index][2],
                                                                                        all_r[index][3])
                print(target_msg_use)
                self.moveit_pub.publish(target_msg_use)

                ###
                # br = tf2_ros.TransformBroadcaster()

                # # Set up the transform message
                # t = geometry_msgs.msg.TransformStamped()
                # t.header.frame_id = "world"
                # t.child_frame_id = "fake_target"
                # t.transform.translation.x = all_t[index][0]
                # t.transform.translation.y = all_t[index][1]
                # t.transform.translation.z = all_t[index][2]

                # # Set rotation (in quaternion)
                # t.transform.rotation.x = all_r[index][0]
                # t.transform.rotation.y = all_r[index][1]
                # t.transform.rotation.z = all_r[index][2]
                # t.transform.rotation.w = all_r[index][3]

                # # Publish the transformation at a set rate
                # rate = rospy.Rate(10)  # 10 Hz
                # while not rospy.is_shutdown():
                #     t.header.stamp = rospy.Time.now()
                #     br.sendTransform(t)
                #     rate.sleep()
            else:
                print("detecting irregular command {}".format(query))   
                msg = String()
                msg.data = query
                self.tactile_pub.publish(msg)
                
                        

if __name__ == '__main__':
    try:
        rospy.init_node('command', anonymous=True)
        PPS_log = PPSLogger()
        PPS_log.ros_tower()
        
    except rospy.ROSInterruptException:
        pass
