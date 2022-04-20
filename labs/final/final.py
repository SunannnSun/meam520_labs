import sys
import numpy as np
from copy import deepcopy

import rospy
import roslib

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# The library you implemented over the course of this semester!
from lib.calculateFK import FK
from lib.calcJacobian import FK
from lib.solveIK import IK
from lib.rrt import rrt
from lib.loadmap import loadmap


def camera_to_robot():
    # Extract homogeneous transformation from the tag 0(t0) to the camera(c)
    T_t0_c = detector.detections[0][1]

    # Extract the relative rotation from the tag 0 to the camera and the position of the tag 0 w.r.t the camera
    R_t0_c = T_t0_c[0:3, 0:3]
    P_t0_c = T_t0_c[0:3, -1]

    # Calculate the relative position of the robot(r) w.r.t. the camera(c)
    # The robot and tag 0 share the same coordinate system, except the tag 0 is 0.5m in the -x direction of the robot;
    # in other words, the robot is 0.5m in the +x direction of the tag 0
    P_r_t0 = np.array([0.5, 0, 0])
    P_r_c = P_t0_c + R_t0_c @ P_r_t0

    # Construct the homogeneous transformation from the camera to the robot
    T_r_c = np.zeros((4, 4))
    T_r_c[0:3, 0:3] = R_t0_c
    T_r_c[:, -1] = np.append(P_r_c, 1)

    # Verify the solution: the rotation matrix is identity matrix, and the position vector is [0, 0, -0.5]
    # print(np.linalg.inv(T_r_c) @ T_t0_c)

    return np.linalg.inv(T_r_c)


def pose_to_joint(block_pose):
    seed = arm.neutral_position()
    q, success, _ = ik.inverse(block_pose, seed)
    print("Success: ", success)
    print("Solution: ", q)
    return q, success


def grasp_and_stack(block_pose, no_blocks):
    tag_pose = block_pose.copy()
    tag_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    while not pose_to_joint(tag_pose)[1]:
        tag_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    above_pose = tag_pose.copy()
    above_pose[2, -1] = above_pose[2, -1] + 0.05
    above_joint, _ = pose_to_joint(above_pose)

    grasp_pose = tag_pose.copy()
    grasp_pose[2, -1] = grasp_pose[2, -1] - 0.025
    grasp_joint, _ = pose_to_joint(grasp_pose)

    stack_up_pose = np.array([[1, 0, 0, 0.56], [0, -1, 0, 0.17], [0, 0, -1, 0.2 + 0.4], [0, 0, 0, 1]])
    stack_up_joint, _ = pose_to_joint(stack_up_pose)

    margin = 0.01
    stack_down_pose = np.array(
        [[1, 0, 0, 0.56], [0, -1, 0, 0.17], [0, 0, -1, 0.2 + no_blocks * 0.05 + 0.025 + margin], [0, 0, 0, 1]])
    stack_down_joint, _ = pose_to_joint(stack_down_pose)
    # _, T_tar = fk.forward(stack_down_joint)
    # print(T_tar)

    print("Move to above the block")
    arm.safe_move_to_position(above_joint)
    print("Move to grasping position")
    arm.safe_move_to_position(grasp_joint)
    print("Close the gripper")
    arm.close_gripper()
    print("Move to above the block")
    arm.safe_move_to_position(above_joint)
    print("Move to above the stacking tower")
    arm.safe_move_to_position(stack_up_joint)
    print("Move to place down the block")
    arm.safe_move_to_position(stack_down_joint)
    print("Release the block")
    arm.open_gripper()
    print("Move to above the stacking tower")
    arm.safe_move_to_position(stack_up_joint)
    print("Move to above the neutral position")
    arm.safe_move_to_position(arm.neutral_position())
    return None


def do_solution_exist(block_pose):
    tag_pose = block_pose.copy()
    tag_pose[0:3, 0:3] = tag_pose[0:3, 0:3] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    pose_to_joint(tag_pose)
    return None


if __name__ == "__main__":

    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    ik = IK()
    fk = FK()
    # arm.safe_move_to_position(arm.neutral_position())
    # arm.open_gripper()
    print("neutral position", arm.neutral_position())

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")  # get set!
    print("Go!\n")  # go!

    # STUDENT CODE HERE

    T_c_r = camera_to_robot()
    T_t0_c = detector.detections[0][1]

    # Detect some tags...

    white_top = []
    white_bot = []
    white_sides = []
    tag_list = detector.get_detections()
    for name, pose in tag_list[1: -1]:
        T_b_r = T_c_r @ pose
        if T_b_r[1, -1] <= 0:

            if name == "tag6":
                white_top.append((name, T_b_r))
            elif name == "tag5":
                white_bot.append((name, T_b_r))
            else:
                white_sides.append((name, T_b_r))

    # for i, block in enumerate(static_blocks):
    print(white_sides)
    #     grasp_and_stack(block[1], i)
    grasp_and_stack(white_sides[0][1], 0)
    # END STUDENT CODE
