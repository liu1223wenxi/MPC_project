clear all
close all
clc

nx = 12;
ny = 12;
nu = 4;
nlmpcobj = nlmpc(nx, ny, nu);

% Specify the prediction model state function using the function name
nlmpcobj.Model.StateFcn = "QuadrotorStateFcn";

% Have Jacobians will significantly improves simulation efficiency.
nlmpcobj.Jacobian.StateFcn = @QuadrotorStateJacobianFcn;

% Fix the random generator seed for reproducibility.
rng(0)

% To check that your prediction model functions for nlobj are valid, 
% use validateFcns for a random point in the state-input space.
validateFcns(nlmpcobj,rand(nx,1),rand(nu,1));

%% Set Variable
Ts = 0.1;  % sampling time [s]
p = 5;    % prediction horizon
m = 2;     % control horizon
nlmpcobj.Ts = Ts;
nlmpcobj.PredictionHorizon = p;
nlmpcobj.ControlHorizon = m;

% % Set Control Constraints
nlmpcobj.MV = struct( ...       % Motor u1 u2 u3 u4
    Min={0;-4;-4;-5}, ...          % Control inputs range
    Max={15;10;10;10});
    % RateMin={-2;-2;-2;-2}, ...  % Control input rates of change
    % RateMax={2;2;2;2} ...
    % );

% The default cost function in nonlinear MPC is a standard 
% quadratic cost function suitable for reference tracking and 
% disturbance rejection.
nlmpcobj.Weights.OutputVariables = [1 1 1 1 1 1 0 0 0 0 0 0];

%% Closed-Loop Simulation
% Specify the initial conditions
x = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];

% Nominal control target (average to keep quadrotor floating)
nloptions = nlmpcmoveopt;
nloptions.MVTarget = [0 0 0 0]; 
mv = nloptions.MVTarget;

% Simulation duration in seconds
Duration = 30;

% Display waitbar to show simulation progress
hbar = waitbar(0,"Simulation Progress");

% MV last value is part of the controller state
lastMV = mv;

% Store states for plotting purposes
xHistory = x';
uHistory = lastMV;

% Simulation loop
for k = 1:(Duration/Ts)

    % Set references for previewing
    t = linspace(k*Ts, (k+p-1)*Ts,p);
    yref = QuadrotorReferenceTrajectory(t);
    % yref =  [18; 10; 15; 0; 0; 0; 0; 0; 0; 0; 0; 0];

    % Compute control move with reference previewing
    xk = xHistory(k,:);
    [uk,nloptions,info] = nlmpcmove(nlmpcobj,xk,lastMV,yref',[],nloptions);

    % Store control move
    uHistory(k+1,:) = uk';
    lastMV = uk;

    % Simulate quadrotor for the next control interval (MVs = uk) 
    ODEFUN = @(t,xk) QuadrotorStateFcn(xk,uk);
    [TOUT,XOUT] = ode45(ODEFUN,[0 Ts], xHistory(k,:)');

    % Update quadrotor state
    xHistory(k+1,:) = XOUT(end,:);

    % Update waitbar
    waitbar(k*Ts/Duration,hbar);
end

% Close waitbar 
close(hbar)

%% Visualization and Results
plotQuadrotorTrajectory;

%% Animation
animateQuadrotorTrajectory;