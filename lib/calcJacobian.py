import numpy as np
from lib.calculateFK import FK

def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q: 0 x 7 configuration vector (of joint angles) [q0,q1,q2,q3,q4,q5,q6]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros(shape=(6, 7))
    ## STUDENT CODE GOES HERE

    fk = FK()
    joint_positions, T0e = fk.forward(q)
    # on = np.array(T0e[1,4], T0e[2,4], T0e[3,4]) #End effector coordinates
    axis = fk.get_axis_of_rotation(q)
    on = joint_positions[-1,:]
    #print(on)
    #print(T0e)
    for x in range(0, 7):
        # S = [0, -axis[x,:][2], axis[x,:][1]; axis[x,:][2], 0, -axis[x,:][0]; ]

        J[0:3,x] = np.cross(axis[:, x], (on-joint_positions[x,:]))
        J[3:6,x] = axis[:, x]

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    #q= np.array([0, -1, 2, -2, 0, 1, 1])
    print(np.round(calcJacobian(q),3))
