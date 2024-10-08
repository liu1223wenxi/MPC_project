function [A,B] = QuadrotorStateJacobianFcn(in1,in2)
%QuadrotorStateJacobianFcn
%    [A,B] = QuadrotorStateJacobianFcn(IN1,IN2)

%    This function was generated by the Symbolic Math Toolbox version 24.1.
%    12-Sep-2024 14:15:43

phi_t = in1(4,:);
phi_dot_t = in1(10,:);
psi_t = in1(6,:);
psi_dot_t = in1(12,:);
theta_t = in1(5,:);
theta_dot_t = in1(11,:);
u1 = in2(1,:);
t2 = cos(phi_t);
t3 = cos(psi_t);
t4 = cos(theta_t);
t5 = sin(phi_t);
t6 = sin(psi_t);
t7 = sin(theta_t);
mt1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,u1.*(t2.*t6-t3.*t5.*t7).*-2.0,u1.*(t2.*t3+t5.*t6.*t7).*2.0,t4.*t5.*u1.*-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,t2.*t3.*t4.*u1.*-2.0,t2.*t4.*t6.*u1.*-2.0,t2.*t7.*u1.*-2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,u1.*(t3.*t5-t2.*t6.*t7).*-2.0,u1.*(t5.*t6+t2.*t3.*t7).*-2.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,psi_dot_t.*(1.13e+2./1.3e+2)];
mt2 = [theta_dot_t.*(-2.0./7.9e+1),0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,psi_dot_t.*(-1.07e+2./1.24e+2),0.0,phi_dot_t.*(-2.0./7.9e+1),0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,theta_dot_t.*(-1.07e+2./1.24e+2),phi_dot_t.*(1.13e+2./1.3e+2),0.0];
A = reshape([mt1,mt2],12,12);
if nargout > 1
    B = reshape([0.0,0.0,0.0,0.0,0.0,0.0,t5.*t6.*-2.0-t2.*t3.*t7.*2.0,t3.*t5.*2.0-t2.*t6.*t7.*2.0,t2.*t4.*2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.064516129032258e+1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0e+3./1.3e+1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.219409282700422e+1],[12,4]);
end
end
