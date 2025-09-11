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

cumulative_pos_err = np.array([0.0, 0.0, 0.0])
ee_position_old = np.array([0.0, 0.0, 0.0])
vel_old = np.array([0.0, 0.0, 0.0])

def move(args: Namespace, robot: SingleArmInterface, run=True):
    # time.sleep(2)
    """
    move
    -----
    come from moveL
    """
    global ee_position_goal
    ee_position_goal = robot.computeT_w_e(robot.q).translation

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

    if last_key_pressed == 'w':
        err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif last_key_pressed == 's':
        err_vector = np.array([0, 0, v, 0, 0, 0])
    elif last_key_pressed == 'a':
        err_vector = np.array([0, v, 0, 0, 0, 0])
    elif last_key_pressed == 'd':
        err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif last_key_pressed == 'c':
        err_vector = impedance_control(robot, J)
    else:
        err_vector = np.array([0, 0, 0, 0, 0, 0])

    
    

    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    #print(robot.q)
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

def pid_control(robot):
    
    """
    PID control in cartesian space.
    Using velocity control for now.
    V_d = Kp * e + Kd * e_dot + Km * integral(e)
    """
    global cumulative_pos_err
    global ee_position_goal
    global ee_position_old

    T_w_e = robot.computeT_w_e(robot.q)
    ee_position = T_w_e.translation  # This is a numpy array [x, y, z]

    ee_position_error =  ee_position_goal - ee_position

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

def impedance_control(robot, J):
    
    """
    """
    global ee_position_goal
    global ee_position_old
    global vel_old

    T_w_e = robot.computeT_w_e(robot.q)
    ee_position = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])

    ee_position_error =  ee_position_goal - ee_position

    dt = robot.dt
    B =  pin.crba(robot.model, robot.data, robot.q)
    B = B[:robot.model.nv, :robot.model.nv]
    M = np.linalg.inv(J @ np.linalg.inv(B) @ J.T)
    M_inv=np.linalg.inv(M)
    f = np.array([0.0, 0.0, 0.0])
    D = 1.4
    K = 1

    vel = vel_old+robot.dt*M_inv@(f-D*vel_old-K*(ee_position_old-ee_position_goal))
    ee_position_old = ee_position.copy()
    vel_old=vel.copy()
    # Pad err_vector with three zeros at the end
    vel = np.concatenate([vel, np.zeros(3)])
    return vel



def key_listener():
    global last_key_pressed
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ['w', 'a', 's', 'd', 'c']:
                last_key_pressed = ch
            elif ch == '\x03':  # Ctrl-C to exit
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def getKeyInputs():
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()

"""
Questions for Yiannis:
1. Should we use velocity control or force control?
There does not seem to be a function to call in the SingleArmInterface to send force or torque commands.
2. If we will be sending velocity commands, how is it possible to implement force control? 
Is it that we are not able to send force commands in the simulation since it is only kinematics sim?
Right now we are using a simple PID controller in cartesian space, and this is because we could not figure out how to 
implement impedance control using velocity control, without first integrating the equation(Me''+De'+Ke=f=>Me'+De+kE=mv,wrong?), which just gives us PID control.
3. How would we know the robots current versus desired end effector position?
Right now we just sum the error over time to get the integral term. (cumulative_err) This leads to stationary errors.
We understand that robot.q gives us the current joint rotations, but how do we get the current end-effector position? Is there a transformation somewhere we do not find?

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""