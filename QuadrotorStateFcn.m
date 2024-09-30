function f = QuadrotorStateFcn(in1,in2)
%QuadrotorStateFcn
%    F = QuadrotorStateFcn(IN1,IN2)

%    This function was generated by the Symbolic Math Toolbox version 24.1.
%    12-Sep-2024 14:15:43

phi_t = in1(4,:);
phi_dot_t = in1(10,:);
psi_t = in1(6,:);
psi_dot_t = in1(12,:);
theta_t = in1(5,:);
theta_dot_t = in1(11,:);
u1 = in2(1,:);
u2 = in2(2,:);
u3 = in2(3,:);
u4 = in2(4,:);
x_dot_t = in1(7,:);
y_dot_t = in1(8,:);
z_dot_t = in1(9,:);
t2 = cos(phi_t);
t3 = cos(psi_t);
t4 = sin(phi_t);
t5 = sin(psi_t);
t6 = sin(theta_t);
f = [x_dot_t;y_dot_t;z_dot_t;phi_dot_t;theta_dot_t;psi_dot_t;u1.*(t4.*t5+t2.*t3.*t6).*-2.0;u1.*(t3.*t4-t2.*t5.*t6).*2.0;t2.*u1.*cos(theta_t).*2.0-9.81e+2./1.0e+2;u2.*8.064516129032258e+1-psi_dot_t.*theta_dot_t.*(1.07e+2./1.24e+2);u3.*(1.0e+3./1.3e+1)+phi_dot_t.*psi_dot_t.*(1.13e+2./1.3e+2);u4.*4.219409282700422e+1-phi_dot_t.*theta_dot_t.*(2.0./7.9e+1)];
end