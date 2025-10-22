from smc import getMinimalArgParser, getRobotFromArgs
from smc.util.define_random_goal import getRandomlyGeneratedGoal
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from phri_control import move
#from real_robot_tests import move
from startup_control import (moveL_only_arm, park_base)
import sys
import argparse
import numpy as np
import pinocchio as pin
import time
import scipy.io as sio
import os
import threading
import sys
import termios
import tty

import debugpy

        
force_pull_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

def key_listener():
    """Continuously listen for keys and call setForceFromKey() accordingly."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # less invasive than setraw()
        print("[INFO] Key listener active. Press Ctrl+C to exit.")
        while True:
            ch = sys.stdin.read(1)  # blocking read of one key
            if ch == 'c':  # Ctrl-C
                print("\n[INFO] Key listener exiting...")
                raise KeyboardInterrupt
            elif ch in ['w', 'a', 's', 'd', 'q', 'e', 'u', 'i', 'o', 'j', 'k', 'l', 'g']:
                setForceFromKey(ch)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("[INFO] Terminal restored.")

def getKeyInputs():
    """Start the key listener in a daemon thread."""
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    print("[INFO] Listener thread started.")


def setForceFromKey(key):
    global f
    force = 10
    torque = 1
    spring_delta_postion = 0.1
    """if key == 'w':
        force_pull_position += np.array([0.0, 0.0, spring_delta_postion, 0.0, 0.0, 0.0])
        #f += np.array([0, 0, force, 0, 0, 0])
        #err_vector = np.array([0, 0, v, 0, 0, 0])
    elif key == 's':
        force_pull_position -= np.array([0.0, 0.0, spring_delta_postion, 0.0, 0.0, 0.0])
        #f += np.array([0, 0, -force, 0, 0, 0])
        #err_vector = np.array([0, 0, -v, 0, 0, 0])
    elif key == 'a':
        force_pull_position += np.array([0.0, spring_delta_postion, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([0, force, 0, 0, 0, 0])
        #err_vector = np.array([0, v, 0, 0, 0, 0])
    elif key == 'd':
        force_pull_position -= np.array([0.0, spring_delta_postion, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([0, -force, 0, 0, 0, 0])  
        #err_vector = np.array([0, -v, 0, 0, 0, 0])
    elif key == 'q':
        force_pull_position += np.array([spring_delta_postion, 0.0, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([force, 0, 0, 0, 0, 0])  
        #err_vector = np.array([0, 0, 0, 0, 0, 0])   
    elif key == 'e':
        force_pull_position -= np.array([spring_delta_postion, 0.0, 0.0, 0.0, 0.0, 0.0])
        #f += np.array([-force, 0, 0, 0, 0, 0])  
        #err_vector = np.array([0, 0, 0, 0, 0, 0])  
    elif key == 'c':
        force_pull_position = ee_position_desired_old.copy()"""
    if key == 'a':
        f += np.array([force, 0, 0, 0, 0, 0])
    elif key == 'd':
        f += np.array([-force, 0, 0, 0, 0, 0])
    elif key == 'e':
        f += np.array([0, force, 0, 0, 0, 0])
    elif key == 'q':
        f += np.array([0, -force, 0, 0, 0, 0])
    elif key == 'w':
        f += np.array([0, 0, force, 0, 0, 0])
    elif key == 's':
        f += np.array([0, 0, -force, 0, 0, 0])  
    elif key == 'i':
        f += np.array([0, 0, 0, 0, 0, torque])
    elif key == 'k':
        f += np.array([0, 0, 0, 0, 0, -torque])
    elif key == 'j':
        f += np.array([0, 0, 0, 0, torque, 0])
    elif key == 'l':
        f += np.array([0, 0, 0, 0, -torque, 0])  
    elif key == 'u':
        f += np.array([0, 0, 0, torque, 0, 0])
    elif key == 'o':
        f += np.array([0, 0, 0, -torque, 0, 0])
    
    print("Current force command: ", f)

def generateNoise(standard_Deviation, length):
    return np.random.normal(0, standard_Deviation, length)

def getForce():
    global f
    returnforce = f + generateNoise(0.1, 6)
    return returnforce

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
   
    parking_lot = np.array([-1.5, -1.15, np.deg2rad(00)])
    # parking_lot = np.array([0, 0, 0])
    # parking_lot = np.array([0.5, 0.5, np.deg2rad(0)])
    
    # define the gripper pose for grabbing the handle
    offset = np.array([parking_lot[0], parking_lot[1], 0])

    translation = np.array([-0.6, 0.0, 1]) + offset
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
    #park_base(args, robot, parking_lot)
    #moveL_only_arm(args, robot, handle_pose)
    print("The robot is now ready to be controlled")
    getKeyInputs()
    move(args, robot, getForce) #for simulation of forces
    #move(args, robot) # delay test

    robot.closeGripper()
    robot.openGripper()
    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
    
    sys.exit(0)
