import rclpy
from rclpy.executors import MultiThreadedExecutor
from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.control_loop_manager import ControlLoopManager
from smc.robots.interfaces.single_arm_interface import SingleArmInterface
from smc.robots.implementations.heron_real import get_args, RealHeronRobotManagerNode
from functools import partial
import sys
import argparse
import numpy as np
import pinocchio as pin
import time
import scipy.io as sio
import os
import time

import debugpy

"""def get_args() -> argparse.Namespace:
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
    return args"""

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
    controlLoopFunction.counter+=1

    q = robot.q
    J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.ee_frame_id, pin.ReferenceFrame.LOCAL)
    # Here you can implement any additional control logic if needed
    
    if controlLoopFunction.counter%1000==0:
        if controlLoopFunction.v==0.05:
            controlLoopFunction.v=0.1
        else:
            controlLoopFunction.v=0.05
        controlLoopFunction.start_t=time.time()
        controlLoopFunction.delay_measured=False
    v=controlLoopFunction.v
    if v == 0.1 and robot.v[8] >= 0.1 and controlLoopFunction.delay_measured!=True:
        controlLoopFunction.delay.append(time.time()-controlLoopFunction.start_t)
        print("Average delay: ", np.mean(controlLoopFunction.delay))
        controlLoopFunction.delay_measured=True
    if v == 0.05 and robot.v[8] <= 0.05 and controlLoopFunction.delay_measured!=True:
        controlLoopFunction.delay.append(time.time()-controlLoopFunction.start_t)
        print("Average delay: ", np.mean(controlLoopFunction.delay))
        controlLoopFunction.delay_measured=True



    # Send velocity command to the robot
    robot.sendVelocityCommand(np.array([0,0,0,0,0,0,0,0,v]))

    return breakFlag, save_past_item, log_item

controlLoopFunction.counter=0
controlLoopFunction.v=0.1
controlLoopFunction.delay=[]
controlLoopFunction.delay_measured=True
controlLoopFunction.start_t=0

if __name__ == "__main__":
    rclpy.init(args=None)

    args = get_args()
    """if args.debug:
        print("Waiting for debugger attach. Go to VSCode and do Run>Start debugging")
        debugpy.listen(5678)  # 5678 is the default debug port
        debugpy.wait_for_client()  # Program pauses here until VS Code attaches"""
    args.robot = "heron"
    args.publish_commands = True
    args.sim = False
    args.robot_ip = "192.168.0.4"
    # args.robot = "ur5e"
    #robot = getRobotFromArgs(args)

    args.ik_solver = "keep_distance_nullspace"
    
    args.real=True #----------------------------------------Change -------------------------------------
    args.visualizer=False
    args.plotter = False
    args.max_v_percentage=0.2
    

    modes_and_loops = []
    robot = RealHeronRobotManagerNode(args)
    robot.base2ee = 0.75
    robot._mode = robot.control_mode.whole_body
    robot._step()

    loop = move(args, robot, run=False)
    modes_and_loops.append((robot.control_mode.whole_body, loop))

    # NOTE: at this point you pass the modes_and_loops list
    robot.setModesAndLoops(modes_and_loops)

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.spin()
   
    #parking_lot = np.array([-1.5, -1.15, np.deg2rad(00)])
    
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
    #if args.visualizer:
        #robot.visualizer_manager.sendCommand({"Mgoal": handle_pose})
    #robot.angle_desired = -45
    # time.sleep(5)
    #park_base(args, robot, parking_lot)
    #moveL_only_arm(args, robot, handle_pose)
    #print('moveL done')
    
    
    #move(args, robot)

    robot.closeGripper()
    robot.openGripper()
    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
    
    sys.exit(0)

