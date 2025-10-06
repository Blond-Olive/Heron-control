load("phri_log.mat")
dt = 0.002
time = 0:dt:(length(f) - 1)*dt

figure(1)
plot(time, f)
title("Force")
xlabel("Time [s]")
ylabel("Force [N]")
legend("x", "y", "z", "rx", "ry", "rz")

figure(2)
plot(time, qs)
title("Joints")
xlabel("Time [s]")
ylabel("Joint rot [rad] / Pos [m]")
legend("Base pos x", "Base pos y", "Base rot x", "Base rot y", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6")

figure(3)
plot(time, v_cmd)
title("Velocity commands")
xlabel("Time [s]")
ylabel("Command")
legend("Base forward", "", "Base rotation", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6")