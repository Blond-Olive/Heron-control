from smc import getMinimalArgParser, getRobotFromArgs
from smc.util.define_random_goal import getRandomlyGeneratedGoal
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from phri_control import move
from startup_control import (moveL_only_arm, park_base)
import sys
import argparse
import numpy as np
import pinocchio as pin
import time
import scipy.io as sio
import os

import debugpy


class Adaptive_controller_manager:
    def __init__(self, robot, alpha=1, beta=1, gamma=2000):
        self.robot = robot
        self.robot.v_ee = 0
        # hyper-parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gain_matrix = np.eye(3)*2000
        # target
        self.f_d = np.zeros(3)
        self.v_d = np.pi/40
        # initalize parameter
        self.err_sum = np.zeros(3)
        theta_rad = np.deg2rad(30)
        Ry = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0,                1, 0               ],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])
        Rz = np.array([
                [np.cos(theta_rad), -np.sin(theta_rad), 0],
                [np.sin(theta_rad),  np.cos(theta_rad), 0],
                [0,                 0,                  1]
            ])
        if robot.task == 1:
            self.x_h = Rz@np.array([1, 0, 0])
        if robot.task == 2:
            self.x_h = Ry@np.array([1, 0, 0])
        if robot.task == 3:
            self.x_h = Rz@np.array([0, -1, 0])
        if robot.task == 4:
            self.x_h = Rz@np.array([1, 0, 0])
            # self.k_h = np.array([3, -1, 2])
        # if robot.task == 2:
        #     Rz = np.array([
        #         [np.cos(theta_rad), -np.sin(theta_rad), 0],
        #         [np.sin(theta_rad),  np.cos(theta_rad), 0],
        #         [0,                 0,                  1]
        #     ])
        #     self.x_h = Rz@np.array([1, 0, 0])
        # self.x_h = np.random.randn(3)
        # self.x_h = self.x_h/np.linalg.norm(self.x_h)
        self.k_h = np.array([0.1, 0.1, 0.1])
        self.v_f = np.zeros(3)
        self.v_ref = np.array([1, 0, 0])
        
        self.time = time.perf_counter()
        self.starttime = time.perf_counter()
        self.x_h_history = []
        self.k_history = []
        self.v_ref_history = []
        
    @staticmethod
    def Proj(x):
        return np.eye(3) - np.outer(x, x)
    
    def get_v_ref(self):
        self.get_v_f()
        self.v_ref = self.v_d * self.x_h - self.Proj(self.x_h) @ self.v_f
        self.get_x_h()
        return self.v_ref
    
    def get_v_f(self):
        # TODO f needs to be in the ee frame
        f = self.robot.getWrench()
        f = f[:3]
        # print(f)
        f_error = f - self.f_d
        self.err_sum += self.Proj(self.x_h) @ f_error 
        self.v_f = self.alpha * f_error + self.beta * self.err_sum
    
    def update_time(self):
        self.time = time.perf_counter()
        
    def get_x_h(self):
        # save data
        self.x_h_history.append(self.x_h.copy())
        self.k_history.append(self.k_h.copy())
        
        dt = time.perf_counter() - self.time
        self.time = time.perf_counter()
        
        T_w_e = self.robot.T_w_e
        self.v_ref = -T_w_e.rotation[:, 2] * abs(self.robot.v_ee)
        vf = self.v_ref
        self.v_ref_history.append((-T_w_e.rotation[:, 2]).copy())
        
        v_ref_norm = np.sign(np.dot(self.x_h.T, self.v_ref)) * np.linalg.norm(vf)

        k_dot = self.gain_matrix @ np.cross(self.x_h, vf)
        self.k_h = self.k_h + dt * k_dot

        x_h_dot = self.gamma * v_ref_norm * self.Proj(self.x_h) @ vf - v_ref_norm * np.cross(self.x_h, self.k_h)
        self.x_h = self.x_h + dt * x_h_dot
        self.x_h = self.x_h / np.linalg.norm(self.x_h)

        # print(self.time - self.starttime)
        if len(self.x_h_history) == 6e3:
            self.save_history_to_mat("es_2_.mat")
        return self.x_h

    def save_history_to_mat(self, filename):
        sio.savemat(filename, {
            "x_h_oe_history": np.array(self.x_h_history),
            "k_oe_history": np.array(self.k_history),
            "v_ref_history": np.array(self.v_ref_history)
        })
        print(f"Data are saved in {filename}")
        
def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Run closed loop inverse kinematics \
    of various kinds. Make sure you know what the goal is before you run!"
    parser = getClikArgs(parser)
    parser.add_argument(
        "--randomly-generate-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if true, the target pose is randomly generated, if false you type it target translation in via text input",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if true, the program waits for a debugger to attach",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    if args.debug:
        print("Waiting for debugger attach. Go to VSCode and do Run>Start debugging")
        debugpy.listen(5678)  # 5678 is the default debug port
        debugpy.wait_for_client()  # Program pauses here until VS Code attaches
    args.robot = "heron"
    # args.robot = "ur5e"
    robot = getRobotFromArgs(args)
    args.ik_solver = "keep_distance_nullspace"
    
    args.real=False
    args.visualizer=True
    args.plotter = False
    args.max_v_percentage=0.2
    robot.base2ee = 0.75
    # 1: open a revolving door 2: revolving drawer 3: sliding door 4/5: sliding drawer
   
    parking_lot = np.array([-1.5, -1.15, np.deg2rad(00)])
    # parking_lot = np.array([0, 0, 0])
    # parking_lot = np.array([0.5, 0.5, np.deg2rad(0)])
    
    # define the gripper pose for grabbing the handle
    offset = np.array([parking_lot[0], parking_lot[1], 0])

    translation = np.array([-0.8, 0.0, 1]) + offset
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    handle_pose = pin.SE3(rotation, translation)
    
    # define the gripper pose before reaching the grab pose
    translation = np.array([-0.4, 0, 1.4]) + offset
    pre_handle_pose = pin.SE3(rotation, translation)
    # Mgoal = getRandomlyGeneratedGoal(args)
    robot.handle_pose = handle_pose
    robot.task = 2
    if args.visualizer:
        robot.visualizer_manager.sendCommand({"Mgoal": handle_pose})
    robot.angle_desired = -45
    # time.sleep(5)
    park_base(args, robot, parking_lot)
    moveL_only_arm(args, robot, handle_pose)
    print("The robot is now ready to be controlled")
    move(args, robot)

    robot.closeGripper()
    robot.openGripper()
    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
    
    sys.exit(0)
