function [ xdesired ] = QuadrotorReferenceTrajectory( t )
% This function generates reference signal for nonlinear MPC controller
% used in the quadrotor path following example.

% Copyright 2019 The MathWorks, Inc.

%#codegen
%circle 1 
% r = 2;
% % r = 1;
% x = r * cos(t);
% y = r * sin(t);
% % z = 4;
% z = 2;

%ellipse 0
a = 6; % Semi-major axis
b = 3; % Semi-minor axis
x = a * cos(0.1*t);
y = b * sin(0.1*t);
z = 2;

%Rose Curve 2
% % 0.1; 0.2
% k = 5; % Number of petals (odd k gives k petals, even k gives 2*k petals)
% r = 5 * sin(k * 0.2*t);
% x = r .* cos(0.2*t);
% y = r .* sin(0.2*t);
% % z = 4;
% z = 2;

%Lissajous Curve 3
% % 0.1; 0.3
% a = 5;
% b = 3;
% delta = pi/2; % Phase difference
% x = a * sin(3 * 0.1*t);
% y = b * sin(2 * 0.1*t + delta);
% % z = 4;
% z = 2; 

% Fig 8 3D   4
% x =6*sin(t/3);
% y = -6*sin(t/3).*cos(t/3);
% z = 6*cos(t/3);

%S-Shaped Path Test
% x = 10 * sin(0.5*t);
% y = 5 * tanh(0.5*0.5 * t);
% z = 2; % Constant height

phi = zeros(1,length(t));
theta = zeros(1,length(t));
psi = zeros(1,length(t));
xdot = zeros(1,length(t));
ydot = zeros(1,length(t));
zdot = zeros(1,length(t));
phidot = zeros(1,length(t));
thetadot = zeros(1,length(t));
psidot = zeros(1,length(t));

xdesired = [x;y;z;phi;theta;psi;xdot;ydot;zdot;phidot;thetadot;psidot];
end

