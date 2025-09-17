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

last_key_pressed = ''  # Global variable to store the last key pressed

ee_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


D = np.diag([100, 100, 100, 1, 1, 0.2]) # last three creates stationary error
K = np.diag([150, 150, 150, 100, 100, 10])
f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

K_p = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

t = 0
goincircle = False
f_add = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def move(args: Namespace, robot: SingleArmInterface, run=True):
    # time.sleep(2)
    """
    move
    -----
    come from moveL
    """
    global ee_position_desired, ee_position_desired_old, force_pull_position
    
    T_w_e = robot.computeT_w_e(robot.q)
    ee_position_desired = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])
    ee_position_desired_old = ee_position_desired.copy()
    force_pull_position = ee_position_desired.copy()

    global x1, x2, vel_desired
    vel_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # pos reference - position desired = 0
    x2 = vel_desired

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
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q

    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)

    

    global last_key_pressed, cumulative_err, f, f_add, ee_position_desired, ee_position_desired_old, t, vel_desired, goincircle, force_pull_position

    T_w_e = robot.computeT_w_e(robot.q)
    ee_position = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])
    force = 10
    if last_key_pressed == 'w':
        force_pull_position += np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        #f += np.array([0, 0, force, 0, 0, 0])
        #err_vector = np.array([0, 0, v, 0, 0, 0])
    elif last_key_pressed == 's':
        force_pull_position -= np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        #f += np.array([0, 0, -force, 0, 0, 0])
        #err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif last_key_pressed == 'a':
        force_pull_position += np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([0, force, 0, 0, 0, 0])
        #err_vector = np.array([0, v, 0, 0, 0, 0])
    elif last_key_pressed == 'd':
        force_pull_position -= np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([0, -force, 0, 0, 0, 0])  
        #err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif last_key_pressed == 'q':
        force_pull_position += np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([force, 0, 0, 0, 0, 0])  
        #err_vector = np.array([0, 0, 0, 0, 0, 0])   
    elif last_key_pressed == 'e':
        force_pull_position -= np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([-force, 0, 0, 0, 0, 0])  
        #err_vector = np.array([0, 0, 0, 0, 0, 0])  
    elif last_key_pressed == 'g':
        goincircle = not goincircle
    elif last_key_pressed == 'c':
        force_pull_position = ee_position_desired_old.copy()
    elif last_key_pressed == 'i':
        f_add += np.array([0, 0, 0, 0, 0, force])
        #err_vector = np.array([0, 0, v, 0, 0, 0])
    elif last_key_pressed == 'k':
        f_add += np.array([0, 0, 0, 0, 0, -force])
        #err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif last_key_pressed == 'j':
        f_add += np.array([0, 0, 0, 0, force, 0])
        #err_vector = np.array([0, v, 0, 0, 0, 0])
    elif last_key_pressed == 'l':
        f_add += np.array([0, 0, 0, 0, -force, 0])  
        #err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif last_key_pressed == 'u':
        f_add += np.array([0, 0, 0, 0, 0, force])
        #err_vector = np.array([0, 0, 0, 0, 0, 0])   
    elif last_key_pressed == 'o':
        f_add += np.array([0, 0, 0, 0, 0, -force])  
        #err_vector = np.array([0, 0, 0, 0, 0, 0]) 
    

    if(goincircle): 
        degrees_per_second = 30
        radius = 0.1
        inner = robot.dt*np.pi/180 *degrees_per_second
        ee_pos_des_old = ee_position_desired.copy()
        ee_position_desired = ee_position_desired_old + radius * np.array([np.cos(t*inner), np.sin(t*inner), 0, 0, 0, 0])
        vel_desired = radius * np.array([-np.sin(t*inner)*inner/robot.dt, np.cos(t*inner)*inner/robot.dt, 0, 0, 0, 0])
        t += 1
    else:
        ee_position_desired = ee_position_desired_old
        vel_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #f = np.array([0, 0, 0, 0, 0, 0])

    translation = force_pull_position[:3]
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis_pos = pin.SE3(rotation, translation)

    robot.visualizer_manager.sendCommand({"Mgoal": vis_pos})
    
    K_s = 100
    f = -K_s * (ee_position - force_pull_position)

    f += f_add

    last_key_pressed = ''  # reset the key
    err_vector = admittance_control(robot, J)
    base_pos = q[:3]
    # Only use translation part for move_towards
    new_base_vel = move_towards(base_pos, ee_position[:3], 0.33, dt=robot.dt)
    base_vel = np.zeros(6)
    base_vel[:3] = new_base_vel
    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    #print(robot.q)
    v_cmd= np.zeros(robot.nv)
    v_cmd_base = base_only_ik(1e-3, q, J, base_vel, robot)
    v_cmd = manipulator_only_ik(1e-3, q, J, err_vector, robot)
    #v_cmd = simple_ik(1e-3, q, J, err_vector, robot)
    v_cmd[:3] = v_cmd_base[:3]  

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

def base_only_ik(tikhonov_damp, q, J, err_vector, robot):
    # Use only the base columns (assume first 3 are base)
    # Remove the second column if needed (as in the original)
    J = np.delete(J, 1, axis=1)
    J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo_inv @ err_vector
    # Re-insert the removed DoF as zero
    qd = np.insert(qd_task, 1, 0.0)
    qd[3:]=0 # reinsert the removed DoF as zero
    return qd

def manipulator_only_ik(tikhonov_damp, q, J, err_vector, robot):
    # Assume base is first 3 DoFs, manipulator is the rest
    J_manip = J[:, 3:]
    # Damped pseudoinverse for manipulator
    J_manip_pinv = J_manip.T @ np.linalg.inv(J_manip @ J_manip.T + np.eye(J_manip.shape[0]) * tikhonov_damp)
    v_manip = J_manip_pinv @ err_vector
    # Compose full velocity command: zeros for base, manipulator velocities
    v_cmd = np.concatenate([np.zeros(3), v_manip])
    return v_cmd

def admittance_control(robot, J):
    
    """
    """
    global ee_position_desired
    global ee_position

    ee_position_error =  ee_position - ee_position_desired

    dt = robot.dt
    B =  pin.crba(robot.model, robot.data, robot.q)
    # B = B[:robot.model.nv, :robot.model.nv]
    M = np.linalg.inv(J @ np.linalg.inv(B) @ J.T)
    M_inv = np.linalg.inv(M)
    global f, D, K, x1, x2, vel_desired, K_p

    x1_dot = x2
    x2_dot = M_inv@(f-D@x2-K@x1)

    x1 = x1 + x1_dot*dt
    x2 = x2 + x2_dot*dt

    p_reference = x1 + ee_position_desired
    p_dot_reference = x2 + vel_desired

    vel_ref = p_dot_reference - K_p@(ee_position - p_reference)
    
    return vel_ref


def key_listener():
    global last_key_pressed
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ['w', 'a', 's', 'd', 'q', 'e', 'c','u','i','o','j','k','l', 'g']:
                last_key_pressed = ch
            elif ch == '\x03':  # Ctrl-C to exit
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def getKeyInputs():
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()

def move_towards(p, target, k, dt):
    direction = target - p
    dist = np.linalg.norm(direction)
    if dist < 5e-1:  # too close
        return -direction*k  # 
    elif dist < 8e-1:  # already there
        return np.zeros(3)
    step = k * dt  # don’t overshoot
    return direction*k



"""
Questions for Yiannis:

Second column take away and bring back in J?

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""