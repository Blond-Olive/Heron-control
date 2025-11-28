%% Start
dt = 0.002
time = 0:dt:((length(f) - 1)*dt);

figure(1)
plot(time, [-f(:, 3), -f(:, 2), -f(:, 1)])
title("Force")
xlabel("Time [s]")
ylabel("Force [N]")
legend("x", "y", "z")
%% 
baserot = atan2(qs(:, 4), qs(:, 3));
qs_centered = qs - qs(1, :);
joints = [qs_centered(:, 1), baserot, qs_centered(:, 4:10)];
figure(2)
plot(time, joints)
title("Joints")
xlabel("Time [s]")
ylabel("Joint rot [rad] / Pos [m]")
legend("Base pos x", "Base pos y", "Base rot ", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6")
%% 
figure(3)
plot(time, v_cmd)
title("Velocity commands")
xlabel("Time [s]")
ylabel("Command")
legend("Base forward", "Nothing", "Base rotation", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6")

time = 0:dt:((length(x1s) - 1)*dt);

figure(4)
plot(time, x1s)
title("x1: Position")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z")

figure(5)
plot(time, x2s)
title("x2: Speed")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z")

figure(6)
plot(time, x2dots)
title("x2 dots: Acceleration")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z")

figure(7)
plot(time, p_refs)
title("Position reference")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z")

figure(8)
plot(time, p_dot_refs)
title("Position reference dot")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z")

figure(9)
plot(time, vel_refs)
title("Velocity reference (Task space)")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z", "rx", "ry", "rz")

% figure(10)
% plot(time, [ee_positions, ee_positions_desired])
% title("Global End effector positions (Task space)")
% xlabel("Time [s]")
% ylabel("Value")
% legend("x", "y", "z", "rx", "ry", "rz", "dx", "dy", "dz", "drx", "dry", "drz")

figure(10)
plot(time, ee_positions(:, 1:3) - ee_positions(1, 1:3))
title("Relative Global End Effector Positions (Task space)")
xlabel("Time [s]")
ylabel("Distance [m]")
legend("x", "y", "z", "rx", "ry", "rz")

figure(12)
plot(time, force_terms)
title("Simulated force (Force term)")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z", "rx", "ry", "rz")

figure(14)
plot(time, [-f_2(:, 3), -f_2(:, 2), -f_2(:, 1)])
title("Force Filtered")
xlabel("Time [s]")
ylabel("Force [N]")
legend("x", "y", "z", "rx", "ry", "rz")

figure(15)
plot(qs(:, 1), qs(:, 2))
title("Base trajectory")
xlabel("X dir")
ylabel("Y dir")
plot(ee_positions(:, 1), ee_positions(:, 2))
axis equal
legend("Base trajectory", "End effector trajectory")

distance = sqrt((qs(:, 1) - ee_positions(:, 1)).^2 + (qs(:, 2) - ee_positions(:, 2)).^2)
figure(16)
hold on
plot(time, distance)
yline(0.5)
title("Distance between end effector and base")
xlabel("Time [s]")
ylabel("Distance [m]")
hold off

%% FFT
t_start = 0;   % analysis start time (seconds)
t_end = 6;     % analysis end time (seconds)

% Convert times to sample indices
i_start = round(t_start / dt) + 1;
i_end = round(t_end / dt) + 1;

% Select segment of interest from all force channels
f_view = f(i_start:i_end, :);
N = size(f_view, 1);            % Number of time points
Fs = 1 / dt              % Sampling frequency (Hz)
freq = (0:N-1) * Fs / N;   % Frequency vector up to Nyquist

% Apply FFT to each channel
F = fft(f);

% Plot magnitude spectrum for the first half of frequencies (real signals)
figure(13)

labels = ["x", "y", "z", "rx", "ry", "rz"]

for k = 1:2
    subplot(1,2,k);
    hold on
    plot([1 1] * 10, ylim)
    semilogx(freq(1:floor(N/2)), abs(F(1:floor(N/2),k)));
    title(labels(k) + ": FFT of force, between " + t_start + " and " + t_end)
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    hold off
end

