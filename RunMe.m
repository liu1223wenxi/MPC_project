clear all
close all
clc

nx = 12;
ny = 12;
nu = 4;
nlmpcobj = nlmpc(nx, ny, nu);

p = 10;    % prediction horizon
m = 10;     % control horizon
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
Ts = 0.01;

% Specify the prediction model state function using the function name
nlmpcobj.Model.StateFcn = "QuadrotorStateFcn";

% Have Jacobians will significantly improves simulation efficiency.
nlmpcobj.Jacobian.StateFcn = @QuadrotorStateJacobianFcn;

% Fix the random generator seed for reproducibility.
rng(0)

% To check that your prediction model functions for nlobj are valid, 
% use validateFcns for a random point in the state-input space.
validateFcns(nlmpcobj,rand(nx,1),rand(nu,1));

thrust = load('Thrust_NeuralNetworkModelFromRegressionLearnerData.mat');
roll = load('Roll_NeuralNetworkModelFromRegressionLearnerData.mat');
pitch = load('Pitch_NeuralNetworkModelFromRegressionLearnerData.mat');
yaw = load('Yaw_NeuralNetworkModelFromRegressionLearnerData.mat');

% Pick controller mpc= 0, nn = 1, 
Controller = 0;
mdl = 'DroneSim.slx';
open_system(mdl);
% 
simOut = sim(mdl);

%%
opt_mvseq = simOut.opt_mv_seq.Data;
opt_xseq = simOut.opt_x_seq.Data;
opt_yseq = simOut.opt_y_seq.Data;
%%%%%%%%%%%% to excel %%%%%%%%%%%%%%
%% Convert arrays to tables -- use sliding window
data_x = simOut.State;
time_x = data_x.Time;
State = data_x.Data;

data_u = simOut.Control_input;
time_u = data_u.Time;
Control_input = data_u.Data;

stateTable = array2table([time_x State], 'VariableNames',...
    {'time_x', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',...
     'xdot', 'ydot', 'zdot', 'p', 'q', 'r'});  % [x;y;z; roll;pitch;yaw; xdot;ydot;zdot; p;q;r]
controlInputTable = array2table(Control_input, 'VariableNames',...
    {'thrust', 'roll angle', 'pitch angle', 'yaw angle'});  

% emptyColumns = array2table(nan(height(stateTable), 1), 'VariableNames', {'Gap'});
% combinedTable = [stateTable, emptyColumns, controlInputTable];
combinedTable = [stateTable, controlInputTable];

writetable(combinedTable, 'State_Control_60sec.csv');

%% Convert arrays to tables -- use opt control
% data_x = simOut.State;
% time_x = data_x.Time;
% State = data_x.Data;
% 
% data_u = simOut.Control_input;
% time_u = data_u.Time;
% Control_input = data_u.Data;

% stateTable = array2table([time_x State], 'VariableNames',...
%     {'time_x', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',...
%      'xdot', 'ydot', 'zdot', 'p', 'q', 'r'});  % [x;y;z; roll;pitch;yaw; xdot;ydot;zdot; p;q;r]
% controlInputTable = array2table(Control_input, 'VariableNames',...
%     {'thrust', 'roll angle', 'pitch angle', 'yaw angle'});  
% 
% 
% if p ~= m 
%     disp('Both horizons must be equivalent in this case');
% else
%     predict_names = {};
%     for i = 1:p+1
%         predict_names = [predict_names, ...
%             sprintf('x%d', i), sprintf('y%d', i), sprintf('z%d', i), ...
%             sprintf('roll%d', i), sprintf('pitch%d', i), sprintf('yaw%d', i), ...
%             sprintf('xdot%d', i), sprintf('ydot%d', i), sprintf('zdot%d', i), ...
%             sprintf('p%d', i), sprintf('q%d', i), sprintf('r%d', i), ...
%             sprintf('thrust%d', i), sprintf('roll_rad%d', i), sprintf('pitch_rad%d', i), ...
%             sprintf('yaw_rad%d', i)];
%     end
%     pair_data = [];
%     for i = 1:p+1
%         pair_data = [pair_data, ...
%             opt_yseq(:, (i-1)*ny + (1:ny)), ... % Extract 12 columns from opt_yseq for the i-th set
%             opt_mvseq(:, (i-1)*nu + (1:nu))];      % Extract 4 columns from opt_mvseq for the i-th set
%     end
%     predict_table = array2table(pair_data, 'VariableNames', predict_names);
% end
% 
% combinedTable = [stateTable, controlInputTable, predict_table];
% 
% writetable(combinedTable, 'State_Control_60sec_withpredict.csv');

%% Animation
% xHistory = simOut.State.Data;
% time = simOut.State.Time;
% animateQuadrotorTrajectory;