%% Start
load("phri_log.mat")
dt = 0.002
time = 0:dt:((length(f) - 1)*dt);

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

figure(10)
plot(time, [ee_positions, ee_positions_desired])
title("Global End effector positions (Task space)")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z", "rx", "ry", "rz", "dx", "dy", "dz", "drx", "dry", "drz")

figure(12)
plot(time, force_terms)
title("Simulated force (Force term)")
xlabel("Time [s]")
ylabel("Value")
legend("x", "y", "z", "rx", "ry", "rz")

%% FFT
t_start = 0;   % analysis start time (seconds)
t_end = 2;     % analysis end time (seconds)

% Convert times to sample indices
i_start = round(t_start / dt) + 1;
i_end = round(t_end / dt) + 1;

% Select segment of interest from all force channels
f_view = f(i_start:i_end, :);
N = size(f_view, 1);            % Number of time points
Fs = 1 / dt;               % Sampling frequency (Hz)
freq = (0:N-1) * Fs / N;   % Frequency vector up to Nyquist

% Apply FFT to each channel
F = fft(f);

% Plot magnitude spectrum for the first half of frequencies (real signals)
figure(13)

for k = 1:6
    subplot(3,2,k);
    plot(freq(1:floor(N/2)), abs(F(1:floor(N/2),k)));
    title("FFT of force, between " + t_start + " and " + t_end)
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
end

