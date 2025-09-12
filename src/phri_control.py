import numpy as np
from argparse import Namespace
from functools import partial
import pinocchio as pin
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.control_loop_manager import ControlLoopManager
from scipy.spatial.transform import Rotation as R
import os
import sys
import termios
import tty
import threading

last_key_pressed = 'a'  # Global variable to store the last key pressed

cumulative_pos_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ee_position_old = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
vel_old = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ee_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


D = np.diag([100, 100, 100, 1, 1, 0.2]) # last three creates stationary error
K = np.diag([150, 150, 150, 100, 100, 10])
f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def move(args: Namespace, robot: SingleArmInterface, run=True):
    # time.sleep(2)
    """
    move
    -----
    come from moveL
    """
    global ee_position_desired
    
    T_w_e = robot.computeT_w_e(robot.q)
    ee_position_desired = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])

    controlLoop = partial(controlLoopFunction, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    getKeyInputs()
    if run:
        loop_manager.run()
        
    else:
        return loop_manager
    
def controlLoopFunction(robot: SingleArmInterface, new_pose, i):


    global ee_position
    global ee_position_old
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q

    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)

    

    global last_key_pressed
    global cumulative_err

    global f

    T_w_e = robot.computeT_w_e(robot.q)
    ee_position = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])
    force = 10
    if last_key_pressed == 'w':
        #f = np.array([0, 0, force, 0, 0, 0])
        err_vector = np.array([0, 0, v, 0, 0, 0])
    elif last_key_pressed == 's':
        #f = np.array([0, 0, -force, 0, 0, 0])
        err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif last_key_pressed == 'a':
        #f = np.array([0, force, 0, 0, 0, 0])
        err_vector = np.array([0, v, 0, 0, 0, 0])
    elif last_key_pressed == 'd':
        #f = np.array([0, -force, 0, 0, 0, 0])  
        err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif last_key_pressed == 'q':
        f = np.array([0, 0, 0, 0, 0, 0])  
        err_vector = np.array([0, 0, 0, 0, 0, 0])   
    elif last_key_pressed == 'e':
        f = np.array([0, force, 0, 0, 0, 0])  
        err_vector = np.array([0, 0, 0, 0, 0, 0])  
    elif last_key_pressed == 'c':
        err_vector = impedance_control(robot, J)

        #f = np.array([0, 0, 0, 0, 0, 0])

   
    

    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    #print(robot.q)
    ee_position_old = ee_position.copy()
    v_cmd = simple_ik(1e-3, q, J, err_vector, robot)
    robot.sendVelocityCommand(v_cmd)

    return breakFlag, save_past_item, log_item

def simple_ik(
    tikhonov_damp,
    q,
    J,
    err_vector,
    robot,
):
    # Remove the second column if needed (as in the original)
    J = np.delete(J, 1, axis=1)
    # Damped pseudoinverse solution for the primary task
    J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo_inv @ err_vector
    # Re-insert the removed DoF as zero
    qd = np.insert(qd_task, 1, 0.0)
    return qd



def impedance_control(robot, J):
    
    """
    """
    global ee_position_desired
    global ee_position_old
    global vel_old
    global ee_position

    ee_position_error =  ee_position - ee_position_desired

    dt = robot.dt
    B =  pin.crba(robot.model, robot.data, robot.q)
    # B = B[:robot.model.nv, :robot.model.nv]
    M = np.linalg.inv(J @ np.linalg.inv(B) @ J.T)
    M_inv = np.linalg.inv(M)
    global f
    global D
    global K

    vel_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vel_desired_old = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x1 = ee_position - ee_position_desired
    x2 = vel_old - vel_desired

    x1_dot = x2
    x2_dot = M_inv@(f-D@x2-K@x1)

    vel = vel_old + (vel_desired - vel_desired_old) + dt*x2_dot

    vel_old= (ee_position - ee_position_old)/dt
    
    return vel



def key_listener():
    global last_key_pressed
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ['w', 'a', 's', 'd', 'q', 'e', 'c']:
                last_key_pressed = ch
            elif ch == '\x03':  # Ctrl-C to exit
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def getKeyInputs():
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()


def pid_control(robot):
    
    """
    PID control in cartesian space.
    Using velocity control for now.
    V_d = Kp * e + Kd * e_dot + Km * integral(e)
    """
    global cumulative_pos_err
    global ee_position_desired
    global ee_position_old
    


    

    ee_position_error =  ee_position_desired - ee_position

    Kp = 0.5
    Kd = 0.01
    Ki = 0.05
    #cumulative_err += err_vector * robot.dt
    err_vector = Kp * ee_position_error + Ki*cumulative_pos_err - Kd*(ee_position - ee_position_old) / robot.dt
    cumulative_pos_err += ee_position_error
    ee_position_old = ee_position.copy()
    # Pad err_vector with three zeros at the end
    err_vector = np.concatenate([err_vector, np.zeros(3)])
    return err_vector

"""
Questions for Yiannis:
Should vel_old (x_2_k-1) be the actual old velocity or just the copy of vel

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""