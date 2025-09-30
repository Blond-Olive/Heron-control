from smc.control.control_loop_manager import ControlLoopManager
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.interfaces.dual_arm_interface import DualArmInterface
from smc.control.controller_templates.point_to_point import (
    EEP2PCtrlLoopTemplate,
    DualEEP2PCtrlLoopTemplate,
)
from smc.robots.interfaces.force_torque_sensor_interface import (
    ForceTorqueOnSingleArmWrist,
)
from smc.control.cartesian_space.ik_solvers import (
    dampedPseudoinverse,
    getIKSolver,
    QPManipMax
)
from functools import partial
import pinocchio as pin
import numpy as np
from argparse import Namespace
from collections import deque
from typing import Callable

from scipy.io import savemat

q_park = []

def controlLoopClik(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = pin.log6(SEerror).vector
    J = robot.getJacobian()
    # compute the joint velocities based on controller you passed
    # qd = ik_solver(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    if args.ik_solver == "QPManipMax":
        v_cmd = QPManipMax(
            J,
            err_vector,
            robot.computeManipulabilityIndexQDerivative(),
            lb=-1 * robot.max_v,
            ub=robot.max_v,
        )
    else:
        v_cmd = ik_solver(J, err_vector)
    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")

    return v_cmd, {}, {}

def controlLoopClik_only_arm(
    #                       J           err_vec     v_cmd
    ik_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
    T_w_goal: pin.SE3,
    args: Namespace,
    robot: SingleArmInterface,
    t: int,
    past_data: dict[str, deque[np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    controlLoopClik
    ---------------
    generic control loop for clik (handling error to final point etc).
    in some version of the universe this could be extended to a generic
    point-to-point motion control loop.
    """
    T_w_e = robot.T_w_e
    SEerror = T_w_e.actInv(T_w_goal)
    err_vector = pin.log6(SEerror).vector
    J = robot.getJacobian()
    J = J[:, 3:]
    # compute the joint velocities based on controller you passed
    # qd = ik_solver(J, err_vector, past_qd=past_data['dqs_cmd'][-1])
    if args.ik_solver == "QPManipMax":
        v_cmd = QPManipMax(
            J,
            err_vector,
            robot.computeManipulabilityIndexQDerivative(),
            lb=-1 * robot.max_v,
            ub=robot.max_v,
        )
    else:
        v_cmd = ik_solver(J, err_vector)
    if v_cmd is None:
        print(
            t,
            "the controller you chose produced None as output, using dampedPseudoinverse instead",
        )
        v_cmd = dampedPseudoinverse(1e-2, J, err_vector)
    else:
        if args.debug_prints:
            print(t, "ik solver success")
    v_cmd = np.concatenate((np.zeros(3), v_cmd))
    return v_cmd, {}, {}

def controlLoopClik_park(robot, clik_controller, target_pose, i, past_data):
    global q_park
    breakFlag = False
    log_item = {}
    save_past_item = {}
    q = robot.q
    q_park.append(q.copy())
    # print(q)
    v_cmd = clik_controller(q, target_pose)
    # v_cmd = np.array([0,0,0.1,0,0,0,0,0,0])
    # v_cmd = np.zeros(robot.nv)
    # v_cmd[2]=1
    current_error = np.linalg.norm(target_pose-np.array([q[0], q[1], np.arctan2(q[3], q[2])]))
    if current_error < robot.args.goal_error:
        breakFlag = True
        savemat("q_park.mat", {"q_park": np.array(q_park)})
        print("q_park saved")
    robot.sendVelocityCommand(v_cmd)

    log_item = {
        "qs": np.zeros(robot.nq),
        "dqs": np.zeros(robot.nv),
        "dqs_cmd": np.zeros(robot.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {}
    # we're not saving here, but need to respect the API,
    # hence the empty dict
    return breakFlag, save_past_item, log_item

def park_base(
    args: Namespace, robot: SingleArmInterface, target_pose, run=True
) -> None | ControlLoopManager:
    # time.sleep(5)
    # assert type(T_w_goal) == pin.SE3
    controlLoop = partial(controlLoopClik_park, robot, parking_base, target_pose)
    # we're not using any past data or logging, hence the empty arguments
    log_item = {
        "qs": np.zeros(robot.model.nq),
        "dqs": np.zeros(robot.model.nv),
        "dqs_cmd": np.zeros(robot.model.nv),
        "err_norm": np.zeros(1),
    }
    save_past_dict = {
        "dqs_cmd": np.zeros(robot.model.nv),
    }
    loop_manager = ControlLoopManager(
        robot, controlLoop, args, save_past_dict, log_item
    )
    if run:
        loop_manager.run()
    else:
        return loop_manager

def moveL_only_arm(
    args: Namespace, robot: SingleArmInterface, T_w_goal: pin.SE3, run=True
) -> None | ControlLoopManager:
    """
    moveL
    -----
    does moveL.
    send a SE3 object as goal point.
    if you don't care about rotation, make it np.zeros((3,3))
    """
    # time.sleep(2)
    assert type(T_w_goal) == pin.SE3
    ik_solver = getIKSolver(args, robot)
    controlLoop = partial(
        EEP2PCtrlLoopTemplate, ik_solver, T_w_goal, controlLoopClik_only_arm, args, robot
    )
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
    if run:
        loop_manager.run()
    else:
        return loop_manager

def parking_base(q, target_pose):
    """
    Compute the linear velocity (v) and angular velocity (omega) for a differential drive robot.
    
    Parameters:
    - q: Robot state, where q[0] and q[1] represent the x and y coordinates, and q[2], q[3] are quaternion values.
    - target_pose: (x, y, theta) Target position in meters and radians.

    Returns:
    - qd: An array containing the computed velocities.
    """

    # Control gains (adjustable)
    k1 = 1.0  # Linear velocity gain
    k2 = 2.0  # Angular velocity gain
    k3 = 1.0  # Orientation error gain

    # Extract robot's current pose
    x_r, y_r, theta_r = (q[0], q[1], np.arctan2(q[3], q[2]))  
    x_t, y_t, theta_t = (target_pose[0] ,target_pose[1] ,target_pose[2])  

    # Compute the relative position between robot and target
    dx = x_r - x_t
    dy = y_r - y_t
    rho = np.hypot(dx, dy)  # Euclidean distance to the target

    # Calculate the target's bearing angle in the global frame
    theta_target = np.arctan2(dy, dx)

    # Compute the bearing error in the robot's local frame
    gamma = theta_target - theta_r + np.pi  # Adjust for orientation

    # Normalize the angle to (-pi, pi] range
    gamma = (gamma + np.pi) % (2 * np.pi) - np.pi

    # Compute the final orientation error
    delta = gamma + theta_r - theta_t
    delta = (delta + np.pi) % (2 * np.pi) - np.pi  # Normalize again

    # Compute linear velocity
    v = k1 * rho * np.cos(gamma)

    # Compute angular velocity, avoiding division by zero
    if gamma == 0:
        omega = 0  
    else:
        omega = k2 * gamma + k1 * (np.sin(gamma) * np.cos(gamma) / gamma) * (gamma + k3 * delta)

    # Construct the velocity output array
    qd = np.array([v, 0, omega, 0, 0, 0, 0, 0, 0])
    return qd