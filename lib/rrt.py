import numpy as np
import random
# from Lab4.code.lib.detectCollision import detectCollision
# from Lab4.code.lib.loadmap import loadmap
# from copy import deepcopy
# from Lab4.code.lib.calculateFK import FK
from lib.detectCollision import detectCollision
from loadmap import loadmap
from copy import deepcopy
from calculateFK import FK

lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
fk = FK()


# def within_box(joints_pos, box):
#     for joint in joints_pos:
#         if (joint > box[-3:]).any():
#             return False
#         if (joint < box[:3]).any():
#             return False
#     return True

def random_joint():
    q_rand = np.zeros(7)
    for i in range(7):
        q_rand[i] = np.random.uniform(lowerLim[i], upperLim[i])
    q_rand_pos, _ = fk.forward(q_rand)
    return q_rand_pos, q_rand

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path

    path = []
    start_edges = []
    end_edges = []
    start_list = [start]
    end_list = [goal]
    count = 0
    within_box = True
    Box = np.array(map.obstacles)

    connected_start = False
    connected_end = False

    while not np.all(np.array([connected_start, connected_end])):
        # Reset connection flag
        connected_start = False
        connected_end   = False

        # Reset within box flag
        box_flag = False
        box_flag_list = [0]
        # Generate Random Position
        while not all(box_flag_list):
            box_flag_list = []
            q_rand = np.zeros(7)
            for i in range(7):
                q_rand[i] = np.random.uniform(lowerLim[i], upperLim[i])
            q_rand_pos, _ = fk.forward(q_rand)
            # print(q_rand_pos)
            # Counter for termination Criterion
            count += 1
            if count == 1000:
                return path
            # Evaluate if within box
            for box in Box:
                if np.all(q_rand_pos[:, 0] > box[0]) and np.all(q_rand_pos[:, 0] < box[1]):
                    if np.all(q_rand_pos[:, 1] > box[2]) and np.all(q_rand_pos[:, 1] < box[3]):
                        if np.all(q_rand_pos[:, 2] > box[4]) and np.all(q_rand_pos[:, 2] < box[5]):
                            box_flag = False
                            box_flag_list.append(box_flag)
                        else:
                            box_flag = True
                            box_flag_list.append(box_flag)
                    else:
                        box_flag = True
                        box_flag_list.append(box_flag)
                else:
                    box_flag = True
                    box_flag_list.append(box_flag)

        # Find the neareast point in start
        dis_lowest = 100
        for i in start_list:
            dis = np.linalg.norm(q_rand - i)
            if dis > dis_lowest:
                pass
            else:
                dis_lowest = dis
                q_start_nearest = i
        q_start_nearest_pos, _ = fk.forward(q_start_nearest)
        collision_list = []
        for box in Box:
            if np.sum(detectCollision(q_start_nearest_pos, q_rand_pos, box)) == 0:
                collision_list.append(True)
            else:
                collision_list.append(False)
        if all(collision_list):
            start_list.append(q_rand)
            start_edges.append((q_start_nearest, q_rand))
            connected_start = True

        # Fin the neareast point in end

        dis_lowest = 100
        for i in end_list:
            dis = np.linalg.norm(q_rand - i)
            if dis > dis_lowest:
                pass
            else:
                dis_lowest = dis
                q_end_nearest = i
        q_end_nearest_pos, _ = fk.forward(q_end_nearest)
        collision_list = []
        for box in Box:
            if np.sum(detectCollision(q_end_nearest_pos, q_rand_pos, box)) == 0:
                collision_list.append(True)
            else:
                collision_list.append(False)
        if all(collision_list):
            end_list.append(q_rand)
            end_edges.append((q_end_nearest, q_rand))
            connected_end = True



    print("path foudn")
    print(start_edges)

    path.append(start_edges[-1][1])
    to_point = start_edges[-1][0]
    while not np.all(to_point == start):
        # print("start appending!")
        for i in range(len(start_edges)):
            if np.all(start_edges[i][1] == to_point):
                index = i
                path.append(start_edges[index][1])
                to_point = start_edges[index][0]
    path.append(start)
    path.reverse()

    to_point = end_edges[-1][0]
    while not np.all(to_point == goal):
        # print("start appending!")
        for i in range(len(end_edges)):
            if np.all(end_edges[i][1] == to_point):
                index = i
                path.append(end_edges[index][1])
                to_point = end_edges[index][0]
    path.append(goal)

    print(len(start_edges))
    print(len(end_edges))
    print("The path is: ", path)
    return path


if __name__ == '__main__':
    map_struct = loadmap("../maps/map2.txt")
    start = np.array([0, -1, 0, -1, 0, 1.57, 0])
    goal = np.array([0, 1.57, 0, -2.07, -0.3, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
