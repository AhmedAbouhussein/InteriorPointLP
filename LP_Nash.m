clear; close all; clc;

% Problem Parameters:
c = [-1; -2; 0; 0; 0]; % Cost Vector
A = [-2 1 1 0 0;       % Constraint Coefficient Matrix
     -1 2 0 1 0; 
      1 2 0 0 1];
b = [2; 7; 3];         % Constraint Constant Vector

% Method Parameters:
mu0   = 10;   % Initial Barrier Parameter
theta = 1/10; % Barrier Reduction Parameter
tol   = 1e-6; % Duality Gap Tolerance
N     = 10;   % Maximum Number of Iterations

% Initial Guesses:

% From Example:
x0 = [0.5; 0.5; 2.5; 6.5; 1.5];
y0 = [-1; -1; -5];
s0 = [1; 11; 1; 1; 5];
% All Elements Set to One
[x1,s1] = deal(ones(size(x0)));
y1 = ones(size(y0));

%% Solve Problem Using Feasible Primal-Dual Algorithm
clc;

% Using x0, y0, and s0:
[x0f,y0f,s0f,Z0f,W0f,k0f,T0f,G0f,R0f,M0f] ...
    = PDM_f(A,b,c,x0,y0,s0,mu0,theta,tol,N);

% Using x1, y1, and s1:
[x1f,y1f,s1f,Z1f,W1f,k1f,T1f,G1f,R1f,M1f] ...
    = PDM_f(A,b,c,x1,y1,s1,mu0,theta,tol,N);

%% Solve Problem Using Infeasible Primal-Dual Algorithm
clc;

% Using x0, y0, and s0:
[x0i,y0i,s0i,Z0i,W0i,k0i,T0i,G0i,R0i,M0i] ...
    = PDM_i(A,b,c,x0,y0,s0,mu0,theta,tol,N);

% Using x1, y1, and s1:
[x1i,y1i,s1i,Z1i,W1i,k1i,T1i,G1i,R1i,M1i] ...
    = PDM_i(A,b,c,x1,y1,s1,mu0,theta,tol,N);

%% Display Results
clc;

% Create Iteration Information:
K0f = (0:k0f)'; K1f = (0:k1f)'; K0i = (0:k0i)'; K1i = (0:k1i)';

% Gather Method Information:
Sol_Prim   = [Z0f(end); Z1f(end); Z0i(end); Z1i(end)]; % Primal Solutions
Sol_Dual   = [W0f(end); W1f(end); W0i(end); W1i(end)]; % Dual Solutions
Iterations = [k0f; k1f; k0i; k1i];                     % Iteration Numbers
Time       = [T0f; T1f; T0i; T0f];                     % Computation Times

% Display Results in Tables:
Comparison = table(Sol_Prim,Sol_Dual,Iterations,Time,...
               'RowNames',{'Feasible with x0','Feasible with x1',...
                           'Infeasible with x0','Infeasible with x1'})
Results_x0_Feasible   = table(K0f,Z0f,W0f,G0f,R0f,M0f,...
                        'VariableNames',{'k','z','w','xs',...
                                         'xs_min_nmu','mu'})
Results_x1_Feasible   = table(K1f,Z1f,W1f,G1f,R1f,M1f,...
                        'VariableNames',{'k','z','w','xs',...
                                         'xs_min_nmu','mu'})
Results_x0_Infeasible = table(K0i,Z0i,W0i,G0i,R0i,M0i,...
                        'VariableNames',{'k','z','w','xs',...
                                         'xs_min_nmu','mu'})
Results_x1_Infeasible = table(K1i,Z1i,W1i,G1i,R1i,M1i,...
                        'VariableNames',{'k','z','w','xs',...
                                         'xs_min_nmu','mu'})