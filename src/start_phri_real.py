import rclpy
from rclpy.executors import MultiThreadedExecutor
from smc.robots.implementations.heron_real import get_args, RealHeronRobotManagerNode
from phri_control import move, savelogs
from functools import partial
import numpy as np
import sys
import time

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

    args.ik_solver = "keep_distance_nullspace"
    
    args.real=True
    args.visualizer=False
    args.plotter = False
    args.max_v_percentage=0.2
    args.gripper = "rs485"
    

    modes_and_loops = []
    robot = RealHeronRobotManagerNode(args)
    robot.base2ee = 0.75
    robot._mode = robot.control_mode.whole_body
    robot._step()

    robot.openGripper()

    print("Please add plank")
    time.sleep(2)

    robot.gripper.move(250, 255, 255)

    print("sleeping...")
    time.sleep(3)

    print("Calibrating force-torque sensor...")
    robot.wrench_offset = robot.calibrateFT(robot.dt)

    def getForce(robot):
        raw_wrench = robot.wrench
        # Correct the coordinate frame: sensor Y maps to robot Z, sensor Z maps to robot -Y
        corrected_wrench = np.zeros_like(raw_wrench)
        corrected_wrench[0] = raw_wrench[0]  # X unchanged
        corrected_wrench[1] = raw_wrench[2]  # Y = -sensor_Z
        corrected_wrench[2] = -raw_wrench[1]   # Z = sensor_Y
        corrected_wrench[3] = raw_wrench[3]   # Torque X unchanged
        corrected_wrench[4] = raw_wrench[5]  # Torque Y = -sensor_torque_Z
        corrected_wrench[5] = -raw_wrench[4]   # Torque Z = sensor_torque_Y
        return corrected_wrench

    getForceFunction = partial(getForce, robot)
    loop = move(args, robot, getForceFunction, run=False)
    modes_and_loops.append((robot.control_mode.whole_body, loop))

    # NOTE: at this point you pass the modes_and_loops list
    robot.setModesAndLoops(modes_and_loops)

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.spin()

    robot.closeGripper()
    robot.openGripper()
    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        #robot._log_manager.saveLog()
        savelogs()
    
    sys.exit(0)