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
    global ee_position_desired, ee_position_desired_old, force_pull_position, p_reference
    
    T_w_e = robot.computeT_w_e(robot.q)
    ee_position_desired = np.concatenate([T_w_e.translation, pin.log3(T_w_e.rotation)])
    ee_position_desired_old = ee_position_desired.copy()
    force_pull_position = ee_position_desired.copy()
    p_reference = ee_position_desired.copy()

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

    

    global last_key_pressed, cumulative_err, f, f_add, ee_position_desired, ee_position_desired_old, t, vel_desired, goincircle, force_pull_position, p_reference

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
    base_pos = q[:4]
    
    v_cmd = manipulator_only_ik(1e-3, q, J, err_vector, robot)

    if np.linalg.norm(f) > 0.1 and np.linalg.norm([force_pull_position[1] - ee_position_desired_old[1], -(force_pull_position[0] - ee_position_desired_old[0])]) > 0.1:
        #goal_angle = -np.arctan2(f[1], f[0])
        goal_angle = -np.arctan2(force_pull_position[1] - ee_position_desired_old[1], -(force_pull_position[0] - ee_position_desired_old[0]))
        current_angle = np.arctan2(q[3], q[2])
        angle_command = 0.5*(goal_angle - current_angle)
        v_cmd[2] = angle_command
        v_cmd[3] -= angle_command

    goal_distance = 0.7 #m
    

    distance = p_reference[:2] - q[:2]
    base_direction_vector = np.array([q[2], q[3]])

    distance_in_base_direction = np.dot(distance, base_direction_vector)

    #distance_error = goal_distance - distance_in_base_direction

    K = 2
    v_cmd[0] = K*distance_in_base_direction
    #v_cmd_base = base_only_ik(1e-3, q, J, err_vector_base, robot)
    
    #v_cmd[:3] = v_cmd_base[:3]  

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
    J_base = J[:, :3]
    # Damped pseudoinverse for base
    J_base_pinv = J_base.T @ np.linalg.inv(J_base @ J_base.T + np.eye(J_base.shape[0]) * tikhonov_damp)
    v_base = J_base_pinv @ err_vector
    # Compose full velocity command: base velocities, zeros for arm
    v_cmd = np.concatenate([v_base, np.zeros(J.shape[1] - 3)])
    return v_cmd

def manipulator_only_ik(tikhonov_damp, q, J, err_vector, robot):
    # Assume base is first 3 DoFs, manipulator is the rest
    J_manip = J[:, 3:]
    # Damped pseudoinverse for manipulator
    J_manip_pinv = J_manip.T @ np.linalg.inv(J_manip @ J_manip.T + np.eye(J_manip.shape[0]) * tikhonov_damp)
    v_manip = J_manip_pinv @ err_vector
    # Compose full velocity command: zeros for base, manipulator velocities
    v_cmd = np.concatenate([np.zeros(3), v_manip])
    return v_cmd

def keep_distance_nullspace(
    tikhonov_damp,
    q,
    J,
    err_vector,
    robot,
    dt: float = 0.01,  # controller cycle time [s]
    Kp_theta: float = 2.0,  # proportional gain for theta
    Ki_theta: float = 0.1,  # integral gain for theta
    integral_limit: float = np.pi,  # anti‑wind‑up saturation
):

    J = np.delete(J, 1, axis=1)
    # -------------------------- Primary task -------------------------- #
    (x_base, y_base, theta_base) = (q[0], q[1], np.arctan2(q[3], q[2]))
    T_w_e = robot.T_w_e
    (x_ee, y_ee) = (T_w_e.translation[0], T_w_e.translation[1])

    # Local‑world‑aligned EE Jacobian and damped pseudo‑inverse
    J_lw = pin.computeFrameJacobian(
        robot.model,
        robot.data,
        q,
        robot.ee_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo @ err_vector  # primary task velocity

    # --------------------- Null‑space: distance task ------------------ #
    d_target = robot.base2ee  # desired base‑EE distance
    dx, dy = x_ee - x_base, y_ee - y_base
    d_current = np.hypot(dx, dy)
    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J  # null‑space projector

    Jx, Jy = J[0, :], J[1, :]
    Jbx, Jby = np.zeros_like(Jx), np.zeros_like(Jx)
    Jbx[0], Jby[0] = q[2], q[3]
    Jd = (dx * (Jx - Jbx) + dy * (Jy - Jby)) / d_current
    z1 = -20.0 * Jd.T * (d_current - d_target)

    # ---------------- Null‑space: heading (theta) task ---------------- #
    z2 = np.zeros_like(z1)

    # EE velocity direction and base heading direction
    xd = (np.delete(J_lw, 1, axis=1) @ qd_task).flatten()
    dir_vee = np.array([xd[0], xd[1]])
    dir_base = np.array([q[2], q[3]])

    def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
        """Signed smallest angle from *b* to *a* (counter‑clockwise positive)."""
        a_n, b_n = a / np.linalg.norm(a), b / np.linalg.norm(b)
        theta_a, theta_b = np.arctan2(a_n[1], a_n[0]), np.arctan2(b_n[1], b_n[0])
        angle = (theta_a - theta_b + np.pi) % (2.0 * np.pi) - np.pi
        # fold >90° to keep the control gentle
        if angle > np.pi / 2:
            angle -= np.pi
        if angle < -np.pi / 2:
            angle += np.pi
        return angle

    theta_err = angle_between_vectors(dir_vee, dir_base)

    # --------------------- theta integral control -------------------- #
    # Persistent (static) accumulator stored on the function object
    if not hasattr(keep_distance_nullspace, "_theta_int"):
        keep_distance_nullspace._theta_int = 0.0
    keep_distance_nullspace._theta_int += theta_err * dt
    # anti‑wind‑up clamping
    keep_distance_nullspace._theta_int = np.clip(
        keep_distance_nullspace._theta_int, -integral_limit, integral_limit
    )

    # Proportional + integral term for theta control
    z2[1] = Kp_theta * theta_err + Ki_theta * keep_distance_nullspace._theta_int

    # --------------------- Combine and return ------------------------ #
    qd_null = N @ (z1 + z2)
    qd = np.insert(qd_task + qd_null, 1, 0.0)  # re‑insert the removed DoF
    return qd


def admittance_control(robot, J):
    
    """
    """
    global ee_position_desired
    global ee_position
    global p_reference

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

def move_towards(p, target, speed, dt):
    p, target = np.array(p, float), np.array(target, float)
    direction = target - p
    dist = np.linalg.norm(direction)
    if dist < 1:  # already there
        return target
    step = min(speed * dt, dist)  # don’t overshoot
    return p + (direction / dist) * step



"""
Questions for Yiannis:

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""