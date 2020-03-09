#!/usr/bin/env python

# -- IMPORT --
import time
import math
import numpy as np
import cv2
# Ros
import rospy
import cv_bridge
import actionlib
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import TwistStamped, Pose
# Ros ACRV
from rv_msgs.msg import ManipulatorState
from rv_msgs.msg import ServoToPoseAction, ServoToPoseGoal
from rv_msgs.msg import MoveToNamedPoseAction, MoveToNamedPoseGoal
from rv_msgs.msg import ActuateGripperAction, ActuateGripperGoal
# Ros YoloHandover
import helper.tf_helpers as tfh
from darknet_ros_msgs.msg import BoundingBoxes


class YoloHandover:

    def __init__(self):

        # Class for preparations
        self.prep = Preparations()
      
        # Parameter config
        self.topic_depth = rospy.get_param('/yolo_handover/subscription/depth')
        self.topic_darknet = rospy.get_param('/yolo_handover/subscription/darknet')
        self.topic_egohands = rospy.get_param('/yolo_handover/subscription/egohands')

        self.topic_arm = rospy.get_param('/yolo_handover/robot/arm_state')
        self.arm_servo_pose = rospy.get_param('/yolo_handover/robot/arm_servo_pose')
        self.arm_named_pose = rospy.get_param('/yolo_handover/robot/arm_named_pose')
        self.arm_gripper = rospy.get_param('/yolo_handover/robot/arm_gripper')
        
        self.dist_ignore = rospy.get_param('/yolo_handover/movement/dist_ignore')
        self.speed_approach = rospy.get_param('/yolo_handover/movement/speed_approach')
        self.scaling_handover = rospy.get_param('/yolo_handover/movement/scaling_handover')
        
        self.gripper_open = rospy.get_param('/yolo_handover/gripper/gripper_open')
        self.gripper_closed = rospy.get_param('/yolo_handover/gripper/gripper_closed')

        self.visualization_type = rospy.get_param('/human_robot_handover_ros/visualization/activated')
        self.visualization_topic = rospy.get_param('/human_robot_handover_ros/visualization/topic')

        # States
        self.state_move = 'startup'
        self.init_state = False
        self.init_depth = False
        self.init_hand = False
        self.init_darknet = False
        self.goal_set = False

        # Variables
        self.depth = None
        self.depth_nan = None
        self.mask_hand = None
        self.best_box = None
        self.last_image_pose = None

        self.servo_goal = ServoToPoseGoal()
        self.error_code = 0
        self.xO = None
        self.yO = None
        self.zO = None
        self.wO = None

        # Init
        self.bridge = cv_bridge.CvBridge()
        self.last_image_pose = None

        # Robot interface
        self.servo_pose_client = actionlib.SimpleActionClient(self.arm_servo_pose, ServoToPoseAction)
        self.servo_pose_client.wait_for_server()
        print("STARTUP -> named_pose_client OK")
        
        self.named_pose_client = actionlib.SimpleActionClient(self.arm_named_pose, MoveToNamedPoseAction)
        self.named_pose_client.wait_for_server()
        print("STARTUP -> named_pose_client OK")

        self.gripper_client = actionlib.SimpleActionClient(self.arm_gripper, ActuateGripperAction)
        self.gripper_client.wait_for_server()
        print("STARTUP -> gripper_client OK")

        # Subsriber
        rospy.Subscriber(self.topic_arm, ManipulatorState, self._callback_state, queue_size=1)
        rospy.Subscriber(self.topic_depth, Image, self._callback_depth, queue_size=1)
        rospy.Subscriber(self.topic_darknet, BoundingBoxes, self._callback_darknet, queue_size=1)
        rospy.Subscriber(self.topic_egohands, CompressedImage, self._callback_egohands, queue_size=1)
        
        # Visualization
        if (self.visualization_type):
            self.pub_visualization = rospy.Publisher(self.visualization_topic, Empty, queue_size=1)
            rospy.Subscriber(self.visualization_topic, Empty, self._callback_visualization, queue_size=1)


    #### FIXED MOVEMENTS ####
    def _move_start(self):
        self.named_pose_client.send_goal(MoveToNamedPoseGoal(pose_name='patros_start', speed=self.speed_approach))
        self.named_pose_client.wait_for_result()

    def _move_home(self):
        self.named_pose_client.send_goal(MoveToNamedPoseGoal(pose_name='patros_home', speed=self.speed_approach))
        self.named_pose_client.wait_for_result()

    def _move_bin(self):
        self.named_pose_client.send_goal(MoveToNamedPoseGoal(pose_name='patros_bin', speed=self.speed_approach))
        self.named_pose_client.wait_for_result()

    def _gripper_open(self, width):
        self.gripper_client.send_goal(ActuateGripperGoal(mode=ActuateGripperGoal.MODE_STATIC, width=width))
        self.gripper_client.wait_for_result()

    def _gripper_close(self, width):
        self.gripper_client.send_goal(ActuateGripperGoal(mode=ActuateGripperGoal.MODE_GRASP, width=width))
        self.gripper_client.wait_for_result()


    #### HELPER FUNCTIONS ####

    # HELPER to reset all parameter
    def _reset_parameter(self):
        self.init_state = False
        self.init_depth = False
        self.init_hand = False
        self.init_darknet = False
        self.goal_set = False

        self.depth = None
        self.depth_nan = None
        self.mask_hand = None
        self.best_box = None
        self.last_image_pose = None

        self.servo_goal = ServoToPoseGoal()
        self.error_code = 0
        self.xO = None
        self.yO = None
        self.zO = None
        self.wO = None

    # HELPER to define and send goal
    def _set_goal(self):

        # Define new goal
        self.servo_goal = ServoToPoseGoal()
        self.servo_goal.stamped_pose.header.frame_id = 'panda_link0'
        self.servo_goal.stamped_pose.pose.position.x = # TODO define where to go in the arm base frame
        self.servo_goal.stamped_pose.pose.position.y = # TODO define where to go in the arm base frame
        self.servo_goal.stamped_pose.pose.position.z = # TODO define where to go in the arm base frame
        self.servo_goal.stamped_pose.pose.orientation.x = self.xO
        self.servo_goal.stamped_pose.pose.orientation.y = self.yO
        self.servo_goal.stamped_pose.pose.orientation.z = self.zO
        self.servo_goal.stamped_pose.pose.orientation.w = self.wO
        self.servo_goal.scaling = self.scaling_handover

        # Send goal
        self.servo_pose_client.send_goal(self.servo_goal)
        self.goal_set = True
        print("MOVE - Goal:")
        print(self.servo_goal.stamped_pose.pose)

        # Publish result
        if (self.visualization_type):
            self.pub_visualization.publish(Empty())


    #### OBJECT-BASED MOVEMENT ####
    def move(self):

        while not rospy.is_shutdown():

            ## HANDLE ERRORS ##
            if (self.error_code != 0):
                rospy.logerr('ERROR CODE: %d' % (self.error_code))

                # Stop any movement
                self.servo_pose_client.cancel_all_goals()

                # Move to start
                self._move_start()
                rospy.sleep(0.5)

                # Eliminate previous measuements (unuseful)
                self._reset_parameter()
                continue
            

            ## STARTUP POSITION ##
            if(self.state_move == 'startup'):

                # Move to start
                self._move_start()
                self.state_move = 'move_ggcnn'

                # Eliminate previous measuements (unuseful)
                self._reset_parameter()


            ## CHECK VIABLE STATE ##
            if not (self.init_state) or not (self.init_depth) or not (self.init_darknet):
                print('State:', self.init_state, 'Depth:', self.init_depth, 'Darknet', self.init_darknet)
                rospy.sleep(0.5)
                continue

            
            ## MOVEMENT ##
            if(self.state_move == 'move_ggcnn'):

                self._set_goal()
                self.servo_pose_client.wait_for_result()
                self.goal_set = False
                print('Goal reached')

                self._gripper_close(0.01)
                self._move_home()
                self._move_bin()
                self._gripper_open(self.gripper_open)

                self._move_start()
                self._reset_parameter()


    #### CALLBACK DEPTH ####
    def _callback_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg)
        self.depth_nan = np.isnan(self.depth).astype(np.uint8)
        self.init_depth = True


    #### CALLBACK EGOHANDS ####
    def _callback_egohands(self, msg):
        self.mask_hand = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.init_hand = True

    
    #### CALLBACK ARM STATE ####
    def _callback_state(self, msg):

        # Save current orientation & error
        self.xO = msg.ee_pose.pose.orientation.x
        self.yO = msg.ee_pose.pose.orientation.y
        self.zO = msg.ee_pose.pose.orientation.z
        self.wO = msg.ee_pose.pose.orientation.w
        self.error_code = msg.errors
        self.init_state = True


    #### CALLBACK DARKNET ROS BOUDNING BOX ####
    def _callback_darknet(self, msg):

        if (self.init_depth):

            depth_nan = self.depth.copy().astype(np.float32)
            depth_nan[self.depth == 0] = np.nan
            depth_nan[self.depth >= self.dist_ignore] = np.nan
            depth_nan[self.mask_hand != 0] = np.nan

            box_list = msg.bounding_boxes
            boxes = []
            for i in range(len(box_list)):
                if(box_list[i].Class != 'person'):
                    depth_mean = np.nanmean(depth_nan[box_list[i].ymin : box_list[i].ymax, box_list[i].xmin : box_list[i].xmax])
                    if(depth_mean > self.dist_ignore) or np.isnan(depth_mean):
                        continue
                    else:
                        boxes.append([box_list[i].xmin, box_list[i].ymin, box_list[i].xmax, box_list[i].ymax, depth_mean])

            if(len(boxes) != 0):
                boxes = np.asarray(boxes)
                self.best_box = boxes[np.argmin(boxes[:,4]),:]
                self.init_darknet = True


    #### CALLBACK VISUALIZATION ####
    def _callback_visualization(self, msg):

        goal_G = Pose()
        goal_G.position.x = self.servo_goal.stamped_pose.pose.position.x
        goal_G.position.y = self.servo_goal.stamped_pose.pose.position.y
        goal_G.position.z = self.servo_goal.stamped_pose.pose.position.z
        goal_G.orientation.x = self.servo_goal.stamped_pose.pose.orientation.x
        goal_G.orientation.y = self.servo_goal.stamped_pose.pose.orientation.y
        goal_G.orientation.z = self.servo_goal.stamped_pose.pose.orientation.z
        goal_G.orientation.w = self.servo_goal.stamped_pose.pose.orientation.w
        tfh.publish_pose_as_transform(goal_G, 'panda_link0', 'GOAL', 0.3)


#### MAIN ####
if __name__ == '__main__':

    rospy.init_node('yolo_handover')
    rthtr = YoloHandover()
    rthtr.move()
    rospy.spin()