clear all
close all
clc

nx = 12;
ny = 12;
nu = 4;
nlmpcobj = nlmpc(nx, ny, nu);

p = 5;    % prediction horizon
m = 2;     % control horizon
nlmpcobj.PredictionHorizon = p;
nlmpcobj.ControlHorizon = m;

% Set Control Constraints
nlmpcobj.MV = struct( ...      
    Min={0;-4;-4;-5}, ...          % Control inputs range
    Max={15;10;10;10});
    % RateMin={-2;-2;-2;-2}, ...  % Control input rates of change
    % RateMax={2;2;2;2} ...
    % );

nlmpcobj.Weights.OutputVariables = [1 1 1 1 1 1 0 0 0 0 0 0];

% Tune Variable R
nlmpcobj.Weights.ManipulatedVariables = [0.1 0.1 0.1 0.1];

Initial = [0 0 0 0 0 0 0 0 0 0 0 0];
mass = 0.5;
Ix = 0.0124;
Iy = 0.0130;
Iz = 0.0237;
Ts = 0.1;

% Specify the prediction model state function using the function name
nlmpcobj.Model.StateFcn = "QuadrotorStateFcn";

% Have Jacobians will significantly improves simulation efficiency.
nlmpcobj.Jacobian.StateFcn = @QuadrotorStateJacobianFcn;

% Fix the random generator seed for reproducibility.
rng(0)

% To check that your prediction model functions for nlobj are valid, 
% use validateFcns for a random point in the state-input space.
validateFcns(nlmpcobj,rand(nx,1),rand(nu,1));

mdl = 'DroneSim.slx';
open_system(mdl);
% 
simOut = sim(mdl);

%%%%%%%%%%%% to excel %%%%%%%%%%%%%%
%% Convert arrays to tables
data_x = simOut.State;
time_x = data_x.Time;
State = data_x.Data;

data_u = simOut.Control_input;
time_u = data_u.Time;
Control_input = data_u.Data;

stateTable = array2table([time_x State], 'VariableNames',...
    {'time_x', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',...
     'xdot', 'ydot', 'zdot', 'p', 'q', 'r'});  % [x;y;z; roll;pitch;yaw; xdot;ydot;zdot; p;q;r]
controlInputTable = array2table([time_u Control_input], 'VariableNames',...
    {'time_u', 'thrust', 'roll angle', 'pitch angle', 'yaw angle'});  

emptyColumns = array2table(nan(height(stateTable), 1), 'VariableNames', {'Gap'});
combinedTable = [stateTable, emptyColumns, controlInputTable];

writetable(combinedTable, 'State_Control_10hr.xlsx', 'Sheet', 'CombinedData');