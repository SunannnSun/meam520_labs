import numpy as np
from lib.calcJacobian import calcJacobian
from numpy import linalg


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q: 0 x 7 vector corresponding to the robot's current configuration.
    :param v: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 0 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
    J = calcJacobian(q_in)
    # zeta = np.array([[v_in[0]], [v_in[1]], [v_in[2]], [omega_in[0]], [omega_in[1]], [omega_in[2]]])
    zeta = (np.append(v_in, omega_in))
    j = 0
    # print(np.shape(J))
    for i in range(0, len(zeta)):
        if np.isnan(zeta[i-j]):
            zeta = np.delete(zeta, i-j, axis=0)
            J = np.delete(J, i-j, axis=0)
            j+=1


    #print(zeta)
    #print(JJ)
    # METHOD 1
    #J_inv = np.dot(np.transpose(J), np.linalg.inv(np.dot(J, np.transpose(J))))
    #dq = np.dot(J_inv, zeta)
    # METHOD 2
    dq = np.linalg.lstsq(J, zeta, rcond = None)[0]
    dq = dq.flatten()


    return dq
