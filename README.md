# Physical Human-Robot Interaction Implementation on the Heron Robot

## Overview (Abstract)

This study investigates human–robot co-manipulation in which a mobile manipulator assists a person in handling and maneuvering a jointly held object. A cooperative control strategy is developed that enables the robot to guide its motion by the human forces transmitted through a wooden board being collaboratively carried. In this framework, the mobile base provides the primary movement while the manipulator maintains object stability and compliance, enabling an intuitive and non-technical interaction.

Both simulations and real-world experiments are conducted to design and validate the control approach on the Heron robot at RobotLab, LTH. The physical properties of the wooden board are identified experimentally, and the influence of noise in the measured interaction forces is characterized and mitigated. An extended Jacobian is formulated to integrate the non-holonomic base with the manipulator dynamics, and the implications of this coupling are systematically analyzed. The method builds on admittance control combined with null-space optimization, allowing simultaneous execution of secondary tasks such as obstacle and singularity avoidance.

The resulting system can co-transport the object while keeping it stable and comfortably positioned for the human partner, demonstrating a promising foundation for cooperative object handling in industrial environments. Proper tuning is required to prevent oscillations in the coupled control loop, introducing a trade-off between responsiveness and perceived inertia. The evaluation of different obstacle-avoidance strategies shows that the framework offers adaptable levels of autonomy that can be matched to the requirements of each task.

**Key Features:**

-   *Admittance control:* An admittance control framwork both with and without spring behavior
    
-   *Whole-Body Motion:* An **extended Jacobian** formulation integrates the base and arm kinematics, enabling simultaneous movement of the mobile base and manipulator for coordinated pushing/pulling of the mechanism.
    
-   *Null-Space Optimization:* Secondary objectives – maintaining a safe distance between the base and arm, obstacle avoidance and multiple others – are achieved via null-space control techniques. This ensures safety (no base-arm collisions) and compensates for the base’s inability to move sideways.
    

Overall, the framework allows for a jointly held wooden board to be cooperativly moved around in translational space, with a minimal effort of the human operator. 


## Usage

1.  **Get SMC**
    
```bash
git clone https://gitlab.control.lth.se/marko-g/ur_simple_control.git
```

2.  **Navigate to ur\_simple\_control/python**
    

```bash
cd ur_simple_control/python
```

3.  **Local install of SMC**
    

```bash
pip install -e .
```

4.  **Install pinocchio through apt**  
    Follow the instructions at [https://stack-of-tasks.github.io/pinocchio/download.html](https://stack-of-tasks.github.io/pinocchio/download.html)  
    Importantly, copy-paste the `export ...` commands under the **"Configure environment variables"** section into your `~/.bashrc`.  
    Then restart the terminal or run:
    

```bash
source ~/.bashrc
```

5.  **Install other Python dependencies**
    

```bash
pip install crocoddyl matplotlib meshcat ur_rtde \
            qpsolvers ecos example_robot_data meshcat_shapes \
            pyqt6 opencv-python qpsolvers quadprog \
            proxsuite casadi pin-pink matplotlib
```
-   **Simulation:** You can run the entire pipeline in a physics-free simulation (kinematic simulation with visualization). Ensure you installed the `smc` package and have a display (MeshCat) for visualization. To simulate a scenario, execute the main control script (for example, `src/mainloop.py`) with Python:
    
    ```bash
    python3 src/start_phri_simulation.py
    ```
    
    By default, this will run in simulation mode (`args.real=False` in the script) and bring up a MeshCat web viewer showing the robot and the mechanism. For force inputs, the terminal is listening for keyboard inputs "wasd + qe". For torque, the same is true but for keys "jkliuo". 
    
-   **Real Robot:** To run experments on the real robot, a ros container running the mir driver is required for base movement. The repository 
    ```bash
    https://git.cs.lth.se/robotlab/ros-containers
    ```

    contains all the necessary details. To accomodate this, all the above steps might also be required to be set up inside of these docker containers.
    
