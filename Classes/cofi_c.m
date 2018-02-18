%  Collaborative filtering class

classdef cofi_c < handle

    properties
        ymu;
        x_l;
        theta_l;
    end
    
    methods
    
        function learn(obj,y,r,max_iter,lambda,x,theta)
            % This function finds the optimal theta for a training set using fminunc
            % Supply x and theta using randn for how many ever features needed
            if nargin < 4; max_iter = 500; end
            if nargin < 5; lambda = 0; end
            
            % set constants
            m = size(x,1);
            nf = size(x,2);
            nc = size(y,2);
            
            y = obj.normalize(y,r);
            
            p = [x(:);theta(:)];
            
            % Set options
            options = optimset('GradObj', 'on', 'MaxIter', max_iter);
            
            % Optimize
            func = @(t)(obj.cost(t,nf,y,r,lambda));
            [p_l, J, exit_flag] = fminunc(func,p,options);
        
            % Roll
            x_l = reshape(p_l(1:m*nf),m,nf);
            theta_l = reshape(p_l(m*nf+1:end),nc,nf);

            obj.x_l = x_l;
            obj.theta_l = theta_l;
            
        end
        
        function [pred] = predict(obj,x_l,theta_l)
        % This function predicts the output using a learned x and theta for all users
            if nargin < 2, x_l = obj.x_l; end
            if nargin < 3, theta_l = obj.theta_l; end
                
            % set constants
            ymu = obj.ymu;

            % Predict
            pred = x_l*theta_l' + ymu;
             
        end
    
        function [ynorm,ymu] = normalize(obj,y,r)  
            % normalize y to have 0 mean rating for each item
            
            % Set constants
            [m, n] = size(y);
            ymu = zeros(m, 1);
            ynorm = zeros(size(y));
            for i = 1:m
                idx = find(r(i, :) == 1);
                ymu(i) = mean(y(i, idx));
                ynorm(i,idx) = y(i, idx) - ymu(i);
            end
            
            obj.ymu = ymu;
        end
    
        function [J, gradient] = cost(obj,p,nf,y,r,lambda)
        % r is the matrix specifying which elements in y should be considered
        % p is a matrix x values first and theta values after
            if nargin < 6, lambda = 0; end
        
            % Set constants
            m = size(y,1);
            nc = size(y,2);
            
            % Get x and theta from p
            x = reshape(p(1:m*nf),m,nf);
            theta = reshape(p(m*nf+1:end),nc,nf);

            J = sum(((x*theta' - y)(r == 1)).^2)/2 + sum(sum(theta.^2))*lambda/2 + sum(sum(x.^2))*lambda/2;

            theta_gradient = ((x*theta' - y).*r)'*x + lambda*theta;
            x_gradient = ((x*theta' - y).*r)*theta + lambda*x;

            gradient = [x_gradient(:); theta_gradient(:)];

        end
    
    end
    
end