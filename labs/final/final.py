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
    print(np.linalg.inv(T_r_c) @ T_t0_c)

    return np.linalg.inv(T_r_c)


def positions_to_joints(block_list):
    joint_list = []
    seed = arm.neutral_position()
    target = np.array([
        [1, 0, 0, 0.3],
        [0, -1, 0, 0.3],
        [0, 0, -1, .25],
        [0, 0, 0, 1],
    ])
    q, success, rollout = ik.inverse(target, seed)

    print("Success: ", success)
    print("Solution: ", q)
    print("Iterations:", len(rollout))
    joint_list.append(q)
    return joint_list


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
    print("neutral position", arm.neutral_position())

    arm.safe_move_to_position(arm.neutral_position())  # on your mark!

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
    for (name, pose) in detector.get_detections():
        print(name, '\n', T_c_r @ pose)

    # Move around...
    # arm.safe_move_to_position(q)

    # END STUDENT CODE
