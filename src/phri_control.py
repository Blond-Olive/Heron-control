import time
import numpy as np
from argparse import Namespace
from functools import partial
import pinocchio as pin
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.control.control_loop_manager import ControlLoopManager
from scipy.spatial.transform import Rotation as R
import os
import threading
import sys
import termios
import tty
import scipy.io as sio

ee_position = np.array([0.0, 0.0, 0.0, 0.0])
f_list=[]

K_rot = np.diag([0, 1, 1]) # Only spring for rotational part
M = np.diag([10, 10, 10, 0.05])  # Mass/inertia for admittance control
# when M was 1, 1, 1 it made oscillations worse

damping_ratio = 2
#D_rot = damping_ratio*2*np.sqrt(M[3:, 3:]@K_rot[3:, 3:])
D = np.diag([20, 20, 20, 4])
W=np.diag([0.25, 1, 1, 1, 1, 1, 1, 1])

K_p = np.diag([3, 3, 3, 3])

t = 0
goincircle = False

obstacle_pos_x = -3.5
obstacle_pos_y = -1.3

logs = {
    "qs": [],
    "dqs": [],
    "v_cmd": [],
    "f": [],
    "f_2": [],
    "x1s": [],
    "x2s": [],
    "x2dots": [],
    "p_refs": [],
    "p_dot_refs": [],
    "vel_refs": [],
    "ee_positions": [],
    "ee_positions_desired": [],
    "force_terms": [],
}

def move(args: Namespace, robot: SingleArmInterface, getForce, run=True):
    # time.sleep(2)
    """
    move
    -----
    come from moveL
    """
    global getForceFunction
    getForceFunction = getForce

    global ee_position_desired, ee_rotation_desired
    
    ee_position_desired = None
    ee_rotation_desired = None

    global x1, x2, vel_desired
    vel_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x1 = np.array([0.0, 0.0, 0.0, 0.0]) # pos reference - position desired = 0
    x2 = vel_desired[:4]

    controlLoop = partial(controlLoopFunction, args, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    robot._log_manager = logger()
    
    args.max_iterations = 1000000  # effectively infinite
    global loop_manager
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
        
    else:
        return loop_manager
    
def controlLoopFunction(args: Namespace, robot: SingleArmInterface, new_pose, i):

    controlLoopFunction.iteration = getattr(controlLoopFunction, 'iteration', 0) + 1

    global ee_position, f_list
    breakFlag = False
    log_item = {}
    save_past_item = {}

    if(controlLoopFunction.iteration < 100):
        return breakFlag, save_past_item, log_item

    q = robot.q

    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)

    global ee_position_desired, ee_rotation_desired, t, vel_desired, goincircle

    T_w_e = robot.computeT_w_e(robot.q)
    
    # Store previous rotation vector for continuity checking
    if not hasattr(controlLoopFunction, 'prev_rot_vec'):
        controlLoopFunction.prev_rot_vec = pin.log3(T_w_e.rotation)

    ee_position = T_w_e.translation.copy()
    ee_rotation = pin.log3(T_w_e.rotation)

    if(ee_position_desired is None):
        ee_position_desired = ee_position.copy()
    if(ee_rotation_desired is None):
        ee_rotation_desired = ee_rotation.copy()
    


    #K_s = 100

    f, f_2 = getForceFunction()  # get the force from the robot or simulation
    f_denoised = denoiseForce(f,2,0.5)
    # Keep force in local frame for end-effector frame admittance control
    #elif controlLoopFunction.iteration % 100 == 0:
        #print("f", f)
    #f = forceFromTorques(f)
    f_denoised = f_denoised[:4]  # ignore torques for now
    #f_local += np.array([-20, 0.0, 0.0, 0.0, 0.0, 0.0])  # bias if needed

    vel_ref = admittance_control(robot, J, f_denoised)

    v_rot = ee_rotation_control(robot)

    vel_ref = np.concatenate([vel_ref, v_rot[1:]])
    
    v_cmd = ik_with_nullspace(1e-3, q, J, vel_ref, robot, f_denoised)
    """f_list.append(f[:3])
    if controlLoopFunction.iteration % 260 == 0 and controlLoopFunction.iteration > 1:
        f_array = np.array(f_list[f_list.__len__()-250:f_list.__len__()])
        f_var = np.var(f_array, ddof=3)
        print("Variance of force over last 250 iterations:", f_var)
        if f_var < 2:
            print("Low variance detected, resseting F/T sensor")
            robot.zeroFtSensor()
        f_list = []"""
    #v_cmd[:3] = v_cmd[:3] * 0.5  # scale base velocities down

    robot.sendVelocityCommand(v_cmd)

    global logs
    logs["qs"].append(q.copy())
    logs["v_cmd"].append(v_cmd.copy())
    logs["f"].append(f.copy())
    logs["f_2"].append(f_denoised.copy())
    logs["vel_refs"].append(vel_ref.copy())
    logs["ee_positions"].append(np.concatenate([ee_position.copy(), ee_rotation.copy()]))
    logs["ee_positions_desired"].append(np.concatenate([ee_position_desired.copy(), ee_rotation_desired.copy()]))
    

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
    W_inv = np.linalg.inv(W)
    #J_pseudo_inv = W_inv@J.T @ np.linalg.inv(J @ W_inv @J.T + np.eye(J.shape[0]) * tikhonov_damp)
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

def ik_with_nullspace(
    tikhonov_damp,
    q,
    J,
    err_vector,
    robot,
    f,
):
    # (Optional) If you also want to remove any direct task-space coupling
    # to base translational directions, you could zero the first 3 rows:
    # if J.shape[0] >= 3:
    #     J[0:3, :] = 0.0
    J = np.delete(J, 1, axis=1)
    W_inv = np.linalg.inv(W)
    #J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    J_pseudo = W_inv@J.T @ np.linalg.inv(J @ W_inv @J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo @ err_vector  # primary task velocity

    z1 = sec_objective_base_distance_to_ee(q, robot, J)
    z2 = sec_objective_rotate_base(q,robot, qd_task, f)
    #z3 = sec_objective_obstacle_avoidance(q, robot, J)
    z4=sec_objective_base_rotate_to_ee(q, robot, J)

    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J  # null‑space projector

    qd_null = N @ (z1+z4)#+ z2) #+ z3)
    #qd = np.insert(qd_task, 1, 0.0)  # re‑insert the removed DoF
    qd = np.insert(qd_task + qd_null, 1, 0.0)
    return qd


def sec_objective_rotate_base(q,robot, qd_task,f,
                            Kp_theta: float = 0.1,  # proportional gain for theta
                            Ki_theta: float = 0.05,  # integral gain for theta
                            integral_limit: float = np.pi):  # anti‑wind‑up saturation limit):
    z2 = np.zeros(8)

    # EE velocity direction and base heading direction
    global ee_position_desired_old, force_pull_position

    T_w_e = robot.computeT_w_e(robot.q)
    f[:3]=T_w_e.rotation @ f[:3]  # transform force to world frame
    dir_force = np.array([f[0], f[1]])
    dir_base = np.array([q[2], q[3]])

    """T_w_e = robot.computeT_w_e(robot.q)
    distance_ee = np.hypot(T_w_e.translation[0] - q[0], T_w_e.translation[1] - q[1])
    dir_ee = np.array([T_w_e.translation[0] - q[0], T_w_e.translation[1] - q[1]])"""

    

    """weight_ee = np.min([1.0, 1.0/50.0 * 1.0/(1.1-distance_ee)**2]) # When distance_vee approaches 1, weight_vee approaches 1
    weight_force = 1.0 - weight_ee

    theta_err_force = angle_between_vectors(dir_force, dir_base)
    theta_err_vee = angle_between_vectors(dir_ee, dir_base)
    theta_err = weight_force * theta_err_force + weight_ee * theta_err_vee"""
    theta_err = angle_between_vectors(dir_force, dir_base)

    # --------------------- theta integral control -------------------- #
    # Persistent (static) accumulator stored on the function object
    if not hasattr(sec_objective_rotate_base, "_theta_int"):
        sec_objective_rotate_base._theta_int = 0.0
    sec_objective_rotate_base._theta_int += theta_err * robot.dt
    # anti‑wind‑up clamping
    sec_objective_rotate_base._theta_int = np.clip(
        sec_objective_rotate_base._theta_int, -integral_limit, integral_limit
    )

    # Proportional + integral term for theta control
    v_rot = Kp_theta * theta_err + Ki_theta * sec_objective_rotate_base._theta_int
    z2[1] = v_rot
    return z2

def sec_objective_base_distance_to_ee(q, robot, J,
                            d_target: float = 0.5,  # desired base‑EE distance [m]
                            Kp_d: float = 40):  # proportional gain for distance
    
    (x_base, y_base, theta_base) = (q[0], q[1], np.arctan2(q[3], q[2]))
    T_w_e = robot.T_w_e
    (x_ee, y_ee) = (T_w_e.translation[0], T_w_e.translation[1])

    d_target = robot.base2ee  # desired base‑EE distance
    dx, dy = x_ee - x_base, y_ee - y_base
    d_current = np.hypot(dx, dy)
    
    Jx, Jy = J[0, :], J[1, :]
    Jbx, Jby = np.zeros_like(Jx), np.zeros_like(Jx)
    Jbx[0], Jby[0] = q[2], q[3]
    Jd = (dx * (Jx - Jbx) + dy * (Jy - Jby)) / d_current
    z1 = -Kp_d * Jd.T * (d_current - d_target)
    return z1

def sec_objective_base_rotate_to_ee(q, robot, J,
                            Kp_r: float = 0.1):  # proportional gain for distance
    
    T_w_e = robot.computeT_w_e(robot.q)
    dir_ee = np.array([T_w_e.translation[0] - q[0], T_w_e.translation[1] - q[1]])
    dir_base = np.array([q[2], q[3]])
    angle_to_ee = np.arctan2(dir_ee[1], dir_ee[0])
    base_heading = np.arctan2(dir_base[1], dir_base[0])
    angle_diff = angle_between_vectors(dir_ee, dir_base)
    z = np.zeros(8) 
    z[1] = Kp_r * angle_diff

    return z

def sec_objective_obstacle_avoidance(q, robot, J,
                                     Kp_d = 2.5, # proportional gain for distance
                                     Kp_r = 1.0, # proportional gain for rotation
                                     distance_threshold = 0.7):  # distance threshold to start avoiding [m]
    global obstacle_pos_x, obstacle_pos_y
    
    (x_base, y_base) = (q[0], q[1])
    
    distance = np.hypot(x_base - obstacle_pos_x, y_base - obstacle_pos_y)
    if distance > distance_threshold:
        return np.zeros(8)
    
    # Repulsive velocity away from the obstacle
    dx, dy = x_base - obstacle_pos_x, y_base - obstacle_pos_y
    Jx, Jy = J[0, :], J[1, :]
    Jbx, Jby = np.zeros_like(Jx), np.zeros_like(Jy)
    Jbx[0], Jby[0] = q[2], q[3]
    Jd = (dx * Jbx + dy * Jby) 
    z = Kp_d * Jd.T / (distance**2)

    # Rotate base away from obstacle
    angle_to_obstacle = np.arctan2(dy, dx)
    base_heading = np.arctan2(q[3], q[2])
    angle_diff = (angle_to_obstacle - base_heading + np.pi) % (2.0 * np.pi) - np.pi
    if angle_diff > 0:
        z[1] -= Kp_r / distance # turn left
    else:
        z[1] += Kp_r / distance  # turn right   

    return z


def admittance_control(robot, J, f_local):
    
    """
    Admittance control in end-effector frame
    Supports two modes:
    - Spring mode (movable_mode=False): Returns to ee_position_desired like a spring
    - Movable mode (movable_mode=True): Stays where moved by forces, no return spring
    """
    global ee_position_desired
    global ee_position

    dt = robot.dt

    global D, K, M, x1, x2, vel_desired, K_p

    # B = B[:robot.model.nv, :robot.model.nv]
    M_inv = np.linalg.inv(M)
    #M = np.linalg.inv(M_inv)

    # Transform positions to end-effector frame for consistent computation
    T_w_e = robot.computeT_w_e(robot.q)
    R_w_e = T_w_e.rotation
    
    # Spring mode: transform desired position to local frame for spring behavior
    ee_position_desired_local = np.zeros(4)#+1 DoF  to give admittnace around x-axis
    ee_position_desired_local[:3] = R_w_e.T @ (ee_position_desired[:3] - T_w_e.translation)
    ee_rotation = pin.log3(T_w_e.rotation)

    ee_rotation_desired_mat = pin.exp3(ee_rotation_desired)
    ee_rotation_desired_local_mat = T_w_e.rotation.T@ee_rotation_desired_mat
    ee_rotation_desired_local = pin.log3(ee_rotation_desired_local_mat)
    ee_position_desired_local[3] = ee_rotation_desired_local[0]
    
    # Transform desired velocity to local frame
    skew_symmetric_tt_vel = skew_symmetric(-R_w_e.T@T_w_e.translation)

    vel_desired_local = np.zeros_like(vel_desired) # !! vel_desired is currently zero!!
    vel_desired_local[:3] = R_w_e.T @ vel_desired[:3]+skew_symmetric_tt_vel@R_w_e.T@vel_desired[3:]
    vel_desired_temp_rot = np.zeros(3)
    vel_desired_temp_rot_local = R_w_e.T @ vel_desired_temp_rot
    vel_desired_local[3] = vel_desired_temp_rot_local[0]
    vel_desired_local = vel_desired_local[:3+1]  # only 4 DoF for now

     


    # Compute admittance dynamics with numerical safety
    # Always include spring term, but K varies by mode (high vs very low stiffness)
    #force_term = f_local - D@x2 - K@x1
    force_term = f_local - D@x2 #- K_rot@x1
    # Check for numerical issues and clip values
    if np.any(np.isnan(force_term)) or np.any(np.isinf(force_term)):
        print("Warning: Invalid force term, resetting to zero")
        force_term = np.zeros_like(force_term)
    
    # Limit maximum force to prevent overflow
    max_force = 1000.0 
    #force_term = np.clip(force_term, -max_force, max_force)
    
    x2_dot = M_inv @ force_term
    
    # Check for numerical issues in acceleration
    if np.any(np.isnan(x2_dot)) or np.any(np.isinf(x2_dot)):
        print("Warning: Invalid acceleration, resetting to zero")
        x2_dot = np.zeros_like(x2_dot)
    
    # Limit maximum acceleration
    #max_accel = 10.0 
    #x2_dot = np.clip(x2_dot, -max_accel, max_accel)

    x1_dot = x2

    x1 = x1 + x1_dot*dt
    x2 = x2 + x2_dot*dt

    
    # Limit velocity to prevent runaway
    max_vel = 5.0  # Adjust this limit as needed  
    #x2 = np.clip(x2, -max_vel, max_vel)

    p_reference_local = x1 + ee_position_desired_local
    p_dot_reference_local = x2 + vel_desired_local

    # Spring mode: position feedback to return to desired position
    position_feedback = K_p @ p_reference_local
    # Check for numerical issues in position feedback
    if np.any(np.isnan(position_feedback)) or np.any(np.isinf(position_feedback)):
        print("Warning: Invalid position feedback, resetting to zero")
        position_feedback = np.zeros_like(position_feedback)

    vel_ref_local = p_dot_reference_local #+ position_feedback

    # Final safety check on output velocity
    if np.any(np.isnan(vel_ref_local)) or np.any(np.isinf(vel_ref_local)):
        print("Warning: Invalid output velocity, resetting to zero")
        vel_ref_local = np.zeros_like(vel_ref_local)
    
    # Limit final output velocity
    max_output_vel = 2.0  # Adjust as needed
    #vel_ref_local = np.clip(vel_ref_local, -max_output_vel, max_output_vel)

    global logs
    logs["x1s"].append(x1.copy())
    logs["x2s"].append(x2.copy())
    logs["x2dots"].append(x2_dot.copy())
    logs["p_refs"].append(p_reference_local.copy())
    logs["p_dot_refs"].append(p_dot_reference_local.copy())
    logs["force_terms"].append(force_term.copy())
    
    if np.linalg.norm(vel_ref_local) > 3.0:
        print("Warning: High output velocity:", vel_ref_local)
        vel_ref_local = np.zeros_like(vel_ref_local)
    if controlLoopFunction.iteration % 750 == 0:
        print("ee_position_desired_local:", ee_position_desired_local)
        print("ee_position:", ee_position)
        print("f_local:", f_local)
        print("vel_ref",vel_ref_local)

    return vel_ref_local

def ee_rotation_control(robot):
    global ee_rotation_desired, K_rot
    
    T_w_e = robot.computeT_w_e(robot.q)
    ee_rotation = pin.log3(T_w_e.rotation)
    ee_rotation = ensure_rot_vec_continuity(ee_rotation, controlLoopFunction.prev_rot_vec)
    controlLoopFunction.prev_rot_vec = ee_rotation.copy()

    ee_rotation_desired_mat = pin.exp3(ee_rotation_desired)
    ee_rotation_desired_local_mat = T_w_e.rotation.T@ee_rotation_desired_mat
    ee_rotation_desired_local = pin.log3(ee_rotation_desired_local_mat)
    
    ee_rotation_local_mat = T_w_e.rotation.T@T_w_e.rotation
    ee_rotation_local = pin.log3(ee_rotation_local_mat)

    rot_err = ee_rotation_local - ee_rotation_desired_local
    v_rot_local = -K_rot @ rot_err

    return v_rot_local

def forceFromTorques(f):
    """
    Translates a force and torque vector in to only forces through an object
    pointing 0.5 m in the local z direction.
    """
    r = np.array([0, 0, 0.5])

    torques = f[3:]

    additional_f = np.cross(torques, r)

    f[3:] = np.zeros_like(f[3:])
    f[:3] += f[:3] + additional_f

    return f


def denoiseForce(f, thresholdForce, thresholdTorque):
    # Low pass filter with cutoff 10Hz
    cutofffreq = 10  # Hz
    dt = 0.002  # assuming control loop at 500Hz
    alpha = 2 * np.pi * cutofffreq * dt / (2 * np.pi * cutofffreq * dt + 1)
    f_denoised = alpha*f + (1-alpha)*denoiseForce.last_f
    denoiseForce.last_f = f_denoised.copy()

    # Zero out components below threshold
    for i in range(3):
        if abs(f[i]) < thresholdForce:
            f_denoised[i] = 0.0
    for i in range(3,6):
        if abs(f[i]) < thresholdTorque:
            f_denoised[i] = 0.0
    return f_denoised

denoiseForce.last_f = np.zeros(6)

def move_towards(p, target, k, dt):
    direction = target - p
    dist = np.linalg.norm(direction)
    if dist < 6e-1:  # too close
        return -direction*k  # 
    elif dist < 9e-1:  # already there
        return np.zeros(3)
    step = k * dt  # don't overshoot
    return direction*k

def ensure_rot_vec_continuity(curr, prev):
    # Return curr (possibly flipped) to maintain continuity with prev
    if np.linalg.norm(curr - prev) > np.pi:
        flipped = -curr
        # keep flipped only if it improves continuity
        if np.linalg.norm(flipped - prev) < np.linalg.norm(curr - prev):
            return flipped
    return curr
    
def global_to_local_vector(vector_global, T_w_e):
    R_w_e = T_w_e.rotation
    vec_local = np.zeros_like(vector_global)
    vec_local[:3] = R_w_e.T @ vector_global[:3]
    vec_local[3:] = R_w_e.T @ vector_global[3:]
    return vec_local

def local_to_global_vector(vector_local, T_w_e):
    """
    Convert a 6D vector (translation[3], rotation_vector[3]) from end-effector local frame
    to world/global frame. R_w_e is the world <- ee rotation matrix (robot.computeT_w_e(...).rotation).
    """
    R_w_e = T_w_e.rotation
    vec_global = np.zeros_like(vector_local)
    # linear part
    vec_global[:3] = R_w_e @ vector_local[:3]
    # angular/rotation-vector part (rotate the axis)
    vec_global[3:] = R_w_e @ vector_local[3:]
    return vec_global

def global_to_local_point(point_global, T_w_e):
    R_w_e = T_w_e.rotation
    t_w_e = T_w_e.translation
    point_local = np.zeros_like(point_global)
    point_local[:3] = R_w_e.T @ (point_global - t_w_e)
    point_local[3:] = R_w_e.T @ point_global[3:]  # rotation vector part unchanged
    return point_local

def local_to_global_point(point_local, T_w_e):
    point_global = np.zeros_like(point_local)
    R_w_e = T_w_e.rotation
    t_w_e = T_w_e.translation
    point_global[:3] = R_w_e @ (point_local[:3] + t_w_e) #TODO Is this correct?
    point_global[3:] = R_w_e @ point_local[3:]  # rotation vector part unchanged    
    return point_global

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
        """Signed smallest angle from *b* to *a* (counter‑clockwise positive)."""
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            #print("Warning: zero-length vector in angle calculation")
            return 0.0
        a_n = a / np.linalg.norm(a)
        b_n = b / np.linalg.norm(b)
        theta_a, theta_b = np.arctan2(a_n[1], a_n[0]), np.arctan2(b_n[1], b_n[0])
        angle = (theta_a - theta_b + np.pi) % (2.0 * np.pi) - np.pi
        # fold >90° to keep the control gentle
        if angle > np.pi / 2:
            angle -= np.pi
        if angle < -np.pi / 2:
            angle += np.pi
        return angle
def skew_symmetric(v):
    """Return the skew-symmetric matrix of a 3D vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def savelogs():
    print("Saving logs to phri_log.mat...")
    global logs
    start_time = time.time_ns()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        matlab_dir = os.path.abspath(os.path.join(base_dir, "..", "matlab"))
        os.makedirs(matlab_dir, exist_ok=True)
        filepath = os.path.join(matlab_dir, "phri_log.mat")
        keys = logs.keys()
        save_dict = {}
        for key in keys:
            save_dict[key] = np.array(logs[key])

        sio.savemat(filepath, {**save_dict})
    except Exception as e:
        print("Warning: failed to save phri_log.mat:", e)
    end_time = time.time_ns()
    print(f"Saved logs to {filepath} in {(end_time - start_time) / 1e9:.4f} seconds ")

class logger: 
    def __init__(self):
        pass

    def saveLog(self):
        savelogs() #This is a HACK

    def storeControlLoopRun(self, log_dict, loop_name, final_iteration):
        pass
"""

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""