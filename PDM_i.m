function [x,y,s,Z,W,k,T,G,R,M] = PDM_i(A,b,c,x0,y0,s0,mu0,theta,tol,N)
% Uses the Infeasible Primal-Dual Method to solve
%          min z=c'*x  s.t.  A*x=b     (Primal Problem)
%          max w=b'*y  s.t.  A'*y+s=c  (Dual Problem)
% INPUT:
%      A = Matrix           = Constraint Coefficient Matrix
%      b = Column Vector    = Constraint Constant Vector
%      c = Column Vector    = Cost Vector
%     x0 = Column Vector    = Initial Guess for Primal Variables
%     y0 = Colume Vector    = Initial Guess for Dual Variables
%     s0 = Column Vector    = Initial Guess for Slack Vector
%    mu0 = Positive Real    = Initial Barrier Parameter
%  theta = Real: 0<theta<1  = Barrier Reduction Parameter
%    tol = Positive Real    = Duality Gap Error Tolerance
%      N = Positive Integer = Maximum Number of Iterations
% OUTPUT:
%      x = Column Vector = Primal Minimizer
%      y = Column Vector = Dual Minimizer
%      s = Column Vector = Slack Vector for Dual Problem
%      Z = Column Vector = Solution History for Primal Problem
%      W = Column Vector = Solution History for Dual Problem
%      k = Integer       = Number of Iterations
%      T = Positive Real = Computation Time
%      G = Column Vector = Duality Gap History
%      R = Column Vector = Complimentary Slackness Residual History
%      M = Column Vector = Barrier Parameter History
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;                % Start Algorithm Timer
    n = length(x0);     % Problem Size
    x = x0;             % Initial Primal Variables
    y = y0;             % Initial Dual Variables
    s = s0;             % Initial Slack Vector
    m = mu0;            % Initial Barrier Parameter   
    e = ones(size(x0)); % Vector with all elements equal to one
    [Z,W,G,R,M] = deal(zeros(N,1)); % Allocate Memory
    Z(1) = c'*x;                    % Store Initial Primal Solution
    W(1) = b'*y;                    % Store Initial Dual Solution
    G(1) = x'*s;                    % Store Initial Duality Gap
    R(1) = G(1)-n*m;                % Store Initial C.S. Residual
    M(1) = NaN;                     % Store Initial Barrier Parameter   
    k=1;                            % In Case G(1) <= tol
    if G(1) > tol                   % If duality gap is large enough
      for k = 1:N                     % Then execute the method
        % Compute Directions
        X  = spdiags(x,0,n,n);    % Diagonal matrix w/ jth diag term = x_j
        S  = spdiags(s,0,n,n);    % Diagonal matrix w/ jth diag term = s_j
        Si = spdiags(1./s,0,n,n); % Calculate Inverse of S
        D = Si*X;                 % Diag Matrix w/ jth diag term = x_j/s_j
        v = m*e-X*S*e;            % Complementary Slackness Residual
        rp = b-A*x;               % Primal Constraint Residual
        rd = c-A'*y-s;            % Dual Constraint Residual
        dy = -(A*D*A')\(A*Si*v-A*D*rd-rp); % Compute Direction of y
        ds = -A'*dy+rd;                    % Compute Direction of s
        dx = Si*v-D*ds;                    % Compute Direction of x

        % Update Estimates
        ix = find(dx<0);               % Indices of negative x components
        alpha_p = min(-x(ix)./dx(ix)); % Calculate Step Length for Primal
        is = find(ds<0);               % Indices of negative s components
        alpha_s = min(-s(is)./ds(is)); % Calculate Step Length for Dual
        alpha_max = min(alpha_p,alpha_s); % Take Maximum Alpha
        alpha = 0.99999*alpha_max;        % Stay below threshold
        if isempty(alpha)                 % If there were no negative x,s
            alpha = 1;                      % Set alpha to 1
        end
        x = x + alpha*dx;    % Update x Variables
        y = y + alpha*dy;    % Update y Variables
        s = s + alpha*ds;    % Update Slack Vector
        
        % Calculate Residuals
        Z(k+1) = c'*x;       % Store Primal Solution
        W(k+1) = b'*y;       % Store Dual Solution
        G(k+1) = x'*s;       % Store Duality Gap
        R(k+1) = G(k+1)-n*m; % Store C.S. Residual
        M(k+1) = m;          % Store Barrier Parameter        
        if G(k+1) <= tol     % If Solution within Tolerance
            break;             % Exit the Loop
        end               
        m = theta*m;         % Reduce Barrier Parameter by a Factor of theta             
      end
    end
    Z = Z(1:k+1); % Cut off Unused Elements of Primal Solution History
    W = W(1:k+1); % Cut off Unused Elements of Dual Solution History
    G = G(1:k+1); % Cut off Unused Elements of Duality Gap History   
    R = R(1:k+1); % Cut off Unused Elements of C.S. Residual History    
    M = M(1:k+1); % Cut off Unused Elements of Barrier Parameter History
    T = toc;      % Record Algorithm Time
end