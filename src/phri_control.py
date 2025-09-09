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
import keyboard  # For key press detection

err_vector = np.array([0, 0, 0, 0, 0, 0])
cumulative_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
last_key_pressed = ''  # Global variable to store the last key pressed

cumulative_pos_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def move(args: Namespace, robot: SingleArmInterface, Adaptive_controller, run=True):
    # time.sleep(2)
    Adaptive_controller.update_time()
    """
    move
    -----
    come from moveL
    """
    

    controlLoop = partial(controlLoopFunction, robot, Adaptive_controller)
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
    
def controlLoopFunction(robot: SingleArmInterface, Adaptive_controller, new_pose, i):

    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q

    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    

    global last_key_pressed
    global cumulative_err

    if last_key_pressed == 'w':
        err_vector = np.array([v, 0, 0, 0, 0, 0])
    elif last_key_pressed == 's':
        err_vector = np.array([-v, 0, 0, 0, 0, 0])
    elif last_key_pressed == 'a':
        err_vector = np.array([0, v, 0, 0, 0, 0])
    elif last_key_pressed == 'd':
        err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif last_key_pressed == 'c':
        err_vector = impedance_control(robot, cumulative_err)
    else:
        err_vector = np.array([0, 0, 0, 0, 0, 0])

    
    cumulative_err += err_vector * robot.dt

    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)
    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    
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
    J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo @ err_vector
    # Re-insert the removed DoF as zero
    qd = np.insert(qd_task, 1, 0.0)
    return qd

def impedance_control(robot, cumulative_err):
    
    """
    impedance control in cartesian space.
    Instead of using velocity control, we use force control, but integrated.
    V_d = Kp * e + Kd * e_dot + Km * integral(e)
    """
    global err_vector
    global cumulative_pos_err
    
    e = err_vector #Using PID for now
    Kp = 0.5
    Kd = 0.1
    Ki = 0.05

    err_vector = -Kp * cumulative_err - Ki*cumulative_pos_err - Kd*e
    print(f"Impedance control err_vector: {err_vector} cumulative_err: {cumulative_err}")
    cumulative_pos_err += cumulative_err
    return err_vector


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
                print(f"Key pressed: {ch}")
            elif ch == '\x03':  # Ctrl-C to exit
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def getKeyInputs():
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
