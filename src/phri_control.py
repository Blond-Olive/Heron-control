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

ee_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

D = None
D_spring = np.diag([100, 100, 100, 5, 5, 0.5]) # last three creates stationary error
D_movable = np.diag([200, 200, 200, 10, 10, 5])  # More moderate damping to avoid numerical issues
K = None
K_spring = np.diag([100, 100, 100, 30, 30, 30])      # Normal stiffness for spring mode
K_movable = np.diag([10, 10, 10, 1, 1, 1])  # Very low stiffness for movable mode

K_p = np.diag([3, 3, 3, 0.5, 0.5, 0.5])

t = 0
goincircle = False
movable_mode = True  # Flag: True = movable mode, False = spring mode (return to desired position)

if(movable_mode):
    D = D_movable
    K = K_movable
else:
    D = D_spring
    K = K_spring

obstacle_pos_x = -3.5
obstacle_pos_y = -1.3

logs = {
    "qs": [],
    "dqs": [],
    "v_cmd": [],
    "f": [],
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

    global ee_position_desired
    
    ee_position_desired = None
    #ee_position_desired_old = ee_position_desired.copy()

    global x1, x2, vel_desired
    vel_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # pos reference - position desired = 0
    x2 = vel_desired

    controlLoop = partial(controlLoopFunction, args, robot)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    args.max_iterations = 1000000  # effectively infinite
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
        
    else:
        return loop_manager
    
def controlLoopFunction(args: Namespace, robot: SingleArmInterface, new_pose, i):

    controlLoopFunction.iteration = getattr(controlLoopFunction, 'iteration', 0) + 1

    global ee_position
    breakFlag = False
    log_item = {}
    save_past_item = {}

    if(controlLoopFunction.iteration < 100):
        return breakFlag, save_past_item, log_item
    if(controlLoopFunction.iteration == 100):
        print("Starting control loop")

    q = robot.q

    v_max = np.pi/40
    # v = np.clip(K * v_max, -v_max, v_max)
    v = v_max
    robot.v_ee = v
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)

    global f, ee_position_desired, t, vel_desired, goincircle

    T_w_e = robot.computeT_w_e(robot.q)
    
    # Store previous rotation vector for continuity checking
    if not hasattr(controlLoopFunction, 'prev_rot_vec'):
        controlLoopFunction.prev_rot_vec = pin.log3(T_w_e.rotation)

    current_rot_vec = pin.log3(T_w_e.rotation)

    current_rot_vec = ensure_rot_vec_continuity(current_rot_vec, controlLoopFunction.prev_rot_vec)

    controlLoopFunction.prev_rot_vec = current_rot_vec
    ee_position = np.concatenate([T_w_e.translation, current_rot_vec])
    
    """if(ee_position_desired is None):
        ee_position_desired = ee_position.copy()
        print("start ee_position_desired", ee_position_desired)
        print("start ee_position", ee_position)
        print("start Twe rot", T_w_e.rotation)"""
    

    """if(goincircle): 
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

    translation = force_pull_position[:3] #np.array([obstacle_pos_x, obstacle_pos_y, 0]) 
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis_pos = pin.SE3(rotation, translation)
    obs_pos=[obstacle_pos_x, obstacle_pos_y, 0.0]

    #robot.visualizer_manager.sendCommand({"obstacle_sphere": (0.5, [obstacle_pos_x, obstacle_pos_y, 0.0])}) #radius, [red, green , blue], 0.5 is one block

    robot.visualizer_manager.sendCommand({"Mgoal": vis_pos})
    
    K_s = 100
    f = -K_s * (ee_position - force_pull_position)
    f += generateNoise(0.1, 6)

    f += f_add
        #f = np.array([0, 0, 0, 0, 0, 0])"""
    
    #K_s = 100

    f = getForceFunction()  # get the force from the robot or simulation
    # Keep force in local frame for end-effector frame admittance control
    
    #f_local += np.array([-20, 0.0, 0.0, 0.0, 0.0, 0.0])  # bias if needed

    vel_ref = admittance_control(robot, J, f)
    
    v_cmd = ik_with_nullspace(1e-3, q, J, vel_ref, robot)

    """if controlLoopFunction.iteration % 200 == 0:
        if np.abs(f[0]) > 5:
            print("pushing in x-direction (red)")
        if np.abs(f[1]) > 5:
            print("pushing in y-direction (green)")
        if np.abs(f[2]) > 5:
            print("pushing in z-direction (blue)")
        #print("vel_ref ", vel_ref[:3])"""
        

    global logs
    logs["qs"].append(q.copy())
    logs["v_cmd"].append(v_cmd.copy())
    logs["f"].append(f.copy())
    if controlLoopFunction.iteration % 1000 == 0:
        sio.savemat("phri_log.mat", {
            "qs": np.array(logs["qs"]),
            "v_cmd": np.array(logs["v_cmd"]),
            "f": np.array(logs["f"]),
        })

    #if controlLoopFunction.iteration % 100 == 0:
        #print("x1 ", x1)


    v_cmd[:3] = np.array([0.0, 0.0, 0.0])
    #v_cmd[8] = 0.0
    #v_cmd=np.array([0,0,0,0,0,0,0,0,0])
        #print("logs saved")
    #base_pos = q[:3]
    # Only use translation part for move_towards
    #new_base_vel = move_towards(base_pos, ee_position[:3], 0.33, dt=robot.dt)
    #base_vel = np.zeros(6)
    #base_vel[:3] = new_base_vel
    # print(J)
    # delete the second columm of Jacobian matrix, cuz y_dot is always 0
    # J[:, 1] = 1e-6
    #print(robot.q)
    #v_cmd= np.zeros(robot.nv)
    #v_cmd_base = base_only_ik(1e-3, q, J, base_vel, robot)
    #v_cmd = manipulator_only_ik(1e-3, q, J, vel_ref, robot)
    #v_cmd = simple_ik(1e-3, q, J, vel_ref, robot)
    #v_cmd[:3] = v_cmd_base[:3]


    """
    if np.linalg.norm(f) > 0.1 and np.linalg.norm([force_pull_position[1] - ee_position_desired_old[1], -(force_pull_position[0] - ee_position_desired_old[0])]) > 0.1:
        #goal_angle = -np.arctan2(f[1], f[0])
        goal_angle = -np.arctan2(force_pull_position[1] - ee_position_desired_old[1], -(force_pull_position[0] - ee_position_desired_old[0]))
        current_angle = np.arctan2(q[3], q[2])
        if goal_angle - current_angle > np.pi/2:
            goal_angle -= np.pi
        elif goal_angle - current_angle < -np.pi/2:
            goal_angle += np.pi
        angle_command = 0.5*(goal_angle - current_angle)
        v_cmd[2] = angle_command
        v_cmd[3] -= angle_command"""


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

def ik_with_nullspace(
    tikhonov_damp,
    q,
    J,
    err_vector,
    robot,
):
    # Remove any control of the base by zeroing the base columns of J.
    # Convention in this code: base DoFs are the first 3 columns (indices 0,1,2).
    # Make a copy so we don't mutate the caller's J unexpectedly.
    J = J.copy()
    n_cols = J.shape[1]
    base_cols = [c for c in (0, 1, 2) if c < n_cols]
    if base_cols:
        J[:, base_cols] = 0.0

    # (Optional) If you also want to remove any direct task-space coupling
    # to base translational directions, you could zero the first 3 rows:
    # if J.shape[0] >= 3:
    #     J[0:3, :] = 0.0
    J = np.delete(J, 1, axis=1)
    J_pseudo = J.T @ np.linalg.inv(J @ J.T + np.eye(J.shape[0]) * tikhonov_damp)
    qd_task = J_pseudo @ err_vector  # primary task velocity

    #z1 = sec_objective_base_distance_to_ee(q, robot, J)
    #z2 = sec_objective_rotate_base(q,robot, qd_task)
    #z3 = sec_objective_obstacle_avoidance(q, robot, J)

    I = np.eye(J.shape[1])
    N = I - J_pseudo @ J  # null‑space projector

    #qd_null = N @ (z1 + z2 + z3)
    qd = np.insert(qd_task, 1, 0.0)#+ qd_null, 1, 0.0)  # re‑insert the removed DoF
    return qd


def sec_objective_rotate_base(q,robot, qd_task,
                            Kp_theta: float = 1.0,  # proportional gain for theta
                            Ki_theta: float = 0.05,  # integral gain for theta
                            integral_limit: float = np.pi):  # anti‑wind‑up saturation)
    z2 = np.zeros(8)

    # EE velocity direction and base heading direction
    global ee_position_desired_old, force_pull_position
    dir_force = np.array([force_pull_position[0] - ee_position_desired_old[0], force_pull_position[1] - ee_position_desired_old[1]])
    dir_base = np.array([q[2], q[3]])

    T_w_e = robot.computeT_w_e(robot.q)
    distance_ee = np.hypot(T_w_e.translation[0] - q[0], T_w_e.translation[1] - q[1])
    dir_ee = np.array([T_w_e.translation[0] - q[0], T_w_e.translation[1] - q[1]])

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


    weight_ee = np.min([1.0, 1.0/50.0 * 1.0/(1.1-distance_ee)**2]) # When distance_vee approaches 1, weight_vee approaches 1
    weight_force = 1.0 - weight_ee

    theta_err_force = angle_between_vectors(dir_force, dir_base)
    theta_err_vee = angle_between_vectors(dir_ee, dir_base)
    theta_err = weight_force * theta_err_force + weight_ee * theta_err_vee

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
                            Kp_d: float = 20.0):  # proportional gain for distance
    
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
    global movable_mode

    dt = robot.dt
    B =  pin.crba(robot.model, robot.data, robot.q)
    # B = B[:robot.model.nv, :robot.model.nv]
    M = np.linalg.inv(J @ np.linalg.inv(B) @ J.T)
    M_inv = np.linalg.inv(M)
    global D, K, x1, x2, vel_desired, K_p

    # Transform positions to end-effector frame for consistent computation
    T_w_e = robot.computeT_w_e(robot.q)
    R_w_e = T_w_e.rotation
    
    if movable_mode:
        # Movable mode: no spring return to original desired position
        ee_position_desired_local = np.zeros(6)  # Stay at current position (local frame origin)
    else:
        # Spring mode: transform desired position to local frame for spring behavior
        ee_position_desired_local = np.zeros_like(ee_position_desired)
        ee_position_desired_local[:3] = R_w_e.T @ (ee_position_desired[:3] - T_w_e.translation)
        
        # For orientation: compute rotation error in local frame
        R_w_e_desired = pin.exp3(ee_position_desired[3:])
        R_w_e_current = T_w_e.rotation
        R_error = R_w_e_current.T @ R_w_e_desired  # Rotation from current to desired in local frame
        ee_position_desired_local[3:] = pin.log3(R_error)
    
    # Transform desired velocity to local frame
    vel_desired_local = np.zeros_like(vel_desired) # !! vel_desired is currently zero!!
    vel_desired_local[:3] = R_w_e.T @ vel_desired[:3]
    vel_desired_local[3:] = R_w_e.T @ vel_desired[3:]

    x1_dot = x2
    
    # Compute admittance dynamics with numerical safety
    # Always include spring term, but K varies by mode (high vs very low stiffness)
    f_local = f_local*2
    force_term = f_local - D@x2 - K@x1
    
    # Check for numerical issues and clip values
    if np.any(np.isnan(force_term)) or np.any(np.isinf(force_term)):
        print("Warning: Invalid force term, resetting to zero")
        force_term = np.zeros_like(force_term)
    
    # Limit maximum force to prevent overflow
    max_force = 1000.0 
    force_term = np.clip(force_term, -max_force, max_force)
    
    x2_dot = M_inv @ force_term
    
    # Check for numerical issues in acceleration
    if np.any(np.isnan(x2_dot)) or np.any(np.isinf(x2_dot)):
        print("Warning: Invalid acceleration, resetting to zero")
        x2_dot = np.zeros_like(x2_dot)
    
    # Limit maximum acceleration
    max_accel = 10.0 
    x2_dot = np.clip(x2_dot, -max_accel, max_accel)

    x1 = x1 + x1_dot*dt
    x2 = x2 + x2_dot*dt
    
    # Limit velocity to prevent runaway
    max_vel = 5.0  # Adjust this limit as needed  
    x2 = np.clip(x2, -max_vel, max_vel)

    p_reference_local = x1 + ee_position_desired_local
    p_dot_reference_local = x2 + vel_desired_local

    if movable_mode:
        # Movable mode: no position feedback, only velocity from admittance dynamics
        vel_ref_local = p_dot_reference_local
    else:
        # Spring mode: position feedback to return to desired position
        position_feedback = K_p @ p_reference_local
        # Check for numerical issues in position feedback
        if np.any(np.isnan(position_feedback)) or np.any(np.isinf(position_feedback)):
            print("Warning: Invalid position feedback, resetting to zero")
            position_feedback = np.zeros_like(position_feedback)
        vel_ref_local = p_dot_reference_local + position_feedback
    
    # Final safety check on output velocity
    if np.any(np.isnan(vel_ref_local)) or np.any(np.isinf(vel_ref_local)):
        print("Warning: Invalid output velocity, resetting to zero")
        vel_ref_local = np.zeros_like(vel_ref_local)
    
    # Limit final output velocity
    max_output_vel = 2.0  # Adjust as needed
    vel_ref_local = np.clip(vel_ref_local, -max_output_vel, max_output_vel)

    #print("p_reference_local ", p_reference_local[:3])
    #print("movable_mode:", movable_mode)
    
    return vel_ref_local

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


"""
Questions for Yiannis:

Second column take away and bring back in J?

v_cmd = [base narrow dir, nothing, <base rot anti-clickwise>, joint 1, joint 2, joint 3 (middle), joint 4, joint 5, joint 6]

q = [base pos (red), base pos (green), <base rot 1, base rot 2>, joint 1, joint 2, joint 3, joint 4, joint 5, joint 6]
"""