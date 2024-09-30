function [cineq, ceq] = droneConstraints(X, U)
    % No equality constraints
    ceq = [];

    % Inequality constraints

    % Extract state variables 
    x = X(1,:);  % x position
    y = X(2,:);  % y position
    z = X(3,:);  % z position
    roll = X(4,:);  % roll angle (phi)
    pitch = X(5,:);  % pitch angle (theta)
    yaw = X(6,:);  % yaw angle (psi)
    xdot = X(7,:);  % x velocity
    ydot = X(8,:);  % y velocity
    zdot = X(9,:);  % z velocity
    p = X(10,:);  % roll rate (phi_dot)
    q = X(11,:);  % pitch rate (theta_dot)
    r = X(12,:);  % yaw rate (psi_dot)

    % Control inputs
    u1 = U(1,:);  % Thrust
    u2 = U(2,:);  % Roll torque
    u3 = U(3,:);  % Pitch torque
    u4 = U(4,:);  % Yaw torque

    % Position constraints
    pose_min_x = -20; pose_max_x = 20;
    pose_min_y = -20; pose_max_y = 20;
    pose_min_z = 0;   pose_max_z = 20;

    % Velocity constraints
    vel_min_x = -3; vel_max_x = 3;
    vel_min_y = -3; vel_max_y = 3;
    vel_min_z = -1; vel_max_z = 3;

    % Angular constraints
    ang_min_roll = deg2rad(-10); ang_max_roll = deg2rad(10);
    ang_min_pitch = deg2rad(-10); ang_max_pitch = deg2rad(10);
    ang_min_yaw = deg2rad(-10); ang_max_yaw = deg2rad(10);

    % Angular rate constraints
    angspd_min_roll = deg2rad(-5); angspd_max_roll = deg2rad(5);
    angspd_min_pitch = deg2rad(-5); angspd_max_pitch = deg2rad(5);
    angspd_min_yaw = deg2rad(-5); angspd_max_yaw = deg2rad(5);

    % Control input constraints
    u1_min = 0; u1_max = 2; % Thrust
    u2_min = -0.5; u2_max = 0.5; % Roll torque
    u3_min = -0.5; u3_max = 0.5; % Pitch torque
    u4_min = -0.5; u4_max = 0.5; % Yaw torque

    % Implement the constraints
    cineq = [...
        x - pose_max_x;  % x <= pose_max_x
        pose_min_x - x;  % x >= pose_min_x
        y - pose_max_y;  % y <= pose_max_y
        pose_min_y - y;  % y >= pose_min_y
        z - pose_max_z;  % z <= pose_max_z
        pose_min_z - z;  % z >= pose_min_z

        xdot - vel_max_x; % xdot <= vel_max_x
        vel_min_x - xdot; % xdot >= vel_min_x
        ydot - vel_max_y; % ydot <= vel_max_y
        vel_min_y - ydot; % ydot >= vel_min_y
        zdot - vel_max_z; % zdot <= vel_max_z
        vel_min_z - zdot; % zdot >= vel_min_z

        roll - ang_max_roll; % roll <= ang_max_roll
        ang_min_roll - roll; % roll >= ang_min_roll
        pitch - ang_max_pitch; % pitch <= ang_max_pitch
        ang_min_pitch - pitch; % pitch >= ang_min_pitch
        yaw - ang_max_yaw; % yaw <= ang_max_yaw
        ang_min_yaw - yaw; % yaw >= ang_min_yaw

        p - angspd_max_roll; % p <= angspd_max_roll
        angspd_min_roll - p; % p >= angspd_min_roll
        q - angspd_max_pitch; % q <= angspd_max_pitch
        angspd_min_pitch - q; % q >= angspd_min_pitch
        r - angspd_max_yaw; % r <= angspd_max_yaw
        angspd_min_yaw - r; % r >= angspd_min_yaw

        u1 - u1_max; % u1 <= u1_max
        u1_min - u1; % u1 >= u1_min
        u2 - u2_max; % u2 <= u2_max
        u2_min - u2; % u2 >= u2_min
        u3 - u3_max; % u3 <= u3_max
        u3_min - u3; % u3 >= u3_min
        u4 - u4_max; % u4 <= u4_max
        u4_min - u4; % u4 >= u4_min
    ];
end
