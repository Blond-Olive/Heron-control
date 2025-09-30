from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.control_loop_manager import ControlLoopManager
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from functools import partial
import sys
import argparse
import numpy as np
import pinocchio as pin
import time
import scipy.io as sio
import os

import debugpy

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

def move(args, robot, run=True):
    """
    Move the robot to the goal pose using closed-loop inverse kinematics.
    """

    controlLoop = partial(controlLoopFunction, robot)
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
    print("move done")
    if run:
        loop_manager.run()
        
    else:
        return loop_manager

def controlLoopFunction(robot: SingleArmInterface, new_pose, i):
    """
    Control loop function to be called in each iteration of the main loop.
    """
    breakFlag = False
    log_item = {}
    save_past_item = {}

    q = robot.q
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)
    # Here you can implement any additional control logic if needed
    v = 0.1
    robot.sendVelocityCommand(np.array([0,0,0,0,0,0,0,0,v]))

    return breakFlag, save_past_item, log_item

if __name__ == "__main__":

    args = get_args()
    print(args)
    if args.debug:
        print("Waiting for debugger attach. Go to VSCode and do Run>Start debugging")
        debugpy.listen(5678)  # 5678 is the default debug port
        debugpy.wait_for_client()  # Program pauses here until VS Code attaches
    args.robot = "heron"
    # args.robot = "ur5e"
    robot = getRobotFromArgs(args)
    args.ik_solver = "keep_distance_nullspace"
    
    args.real=False #----------------------------------------Change -------------------------------------
    args.visualizer=True
    args.plotter = False
    args.max_v_percentage=0.2
    robot.base2ee = 0.75
   
    parking_lot = np.array([-1.5, -1.15, np.deg2rad(00)])
    
    # define the gripper pose for grabbing the handle
    #offset = np.array([parking_lot[0], parking_lot[1], 0])

    #translation = np.array([-0.8, 0.0, 1]) + offset
    #rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #handle_pose = pin.SE3(rotation, translation)
    
    # define the gripper pose before reaching the grab pose
    #translation = np.array([-0.4, 0, 1.4]) + offset
    #pre_handle_pose = pin.SE3(rotation, translation)
    # Mgoal = getRandomlyGeneratedGoal(args)
    #robot.handle_pose = handle_pose
    robot.task = 2
    #if args.visualizer:
        #robot.visualizer_manager.sendCommand({"Mgoal": handle_pose})
    robot.angle_desired = -45
    # time.sleep(5)
    #park_base(args, robot, parking_lot)
    #moveL_only_arm(args, robot, handle_pose)
    #print('moveL done')
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

