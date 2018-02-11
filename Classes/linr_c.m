classdef linr_c < handle

  properties
  
    theta_l; % Learned theta
    mu;
    sigma;
    
  end
  
  methods
    
    function [J, gradient] = cost(obj,x,y,theta,lambda)
      % This function computes the cost function for x, y and theta.
      % It allows regularization
      % theta is a row vector corresponding to x
      if nargin < 4, lambda = 0; end
      
      % set constants
      m = size(x,1);
      
      h = x*theta';
      
      % Calculate cost function
      J = sum((h-y).^2)/2/m + sum(theta(:,2:end).^2)*lambda/2/m;
      
      % Compute gradient
      gradient = (x'*(h-y)/m)';

    end
    
    function [theta_l,mu,sigma,J_history] = learn_grad(obj,x,y,alpha,num_iters,lambda)
    % This function uses machine learning to learn a model for training data
    % x and y using gradient descent

        % set constants
        m = length(x);
        
        % Feature Normalize
        [x,mu,sigma] = Feature_Normalize(x);
        obj.mu = mu;
        obj.sigma = sigma;

        % Add bias
        x = [ones(m,1),x];

        % Initialize theta
        theta = zeros(1,size(x,2));

        % Optimize
        [theta_l,J_history] = Gradient_Descent(x,y,theta,alpha,num_iters,lambda);
        
        obj.theta_l = theta_l;
    end
    
    function [pred] = predict(obj,x,theta_l)
    % This function predicts the output using a learned theta for x
        
        if nargin < 3, theta_l = obj.theta_l; end
            
        % set constants
        m = size(x,1);
        mu = obj.mu;
        sigma = obj.sigma;
        
        % Feature Normalize
        x = Feature_Normalize(x,mu,sigma);
        
        % Add bias
        x = [ones(m,1),x];
        
        % Predict
        pred = x*theta_l';
         
    end
    
    function [theta_l] = normal_solve(obj,x,y)
        % This function obtains the optimal theta for linear regression without gradient descent
        % Note that the theta calculated using the normal equation will be different from
        % the one you get using gradient descent since the normal equation does not use
        % feature normalization

        % set constants
        m = length(x);
        
        % Add bias
        x = [ones(m,1),x];
        
        % Use normal equation to solve
        theta_l = (x'*x)^-1*x'*y;
        theta_l = theta_l(:)';
        obj.mu = 0;
        obj.sigma = 1;

        obj.theta_l = theta_l;

    end
    
    function [theta_l,mu,sigma,J] = learn(obj,x,y,max_iter,lambda)
        % This function finds the optimal theta for a training set using fminunc
        if nargin < 4; max_iter = 500; end
        if nargin < 5; lambda = 0; end

        % set constants
        m = length(x);
        
        % Feature Normalize
        [x,mu,sigma] = Feature_Normalize(x);
        obj.mu = mu;
        obj.sigma = sigma;

        % Add bias
        x = [ones(m,1),x];

        % Initialize theta
        theta = zeros(1,size(x,2));
            
        % Set options
        options = optimset('GradObj', 'on', 'MaxIter', max_iter);
        
        % Optimize
        func = @(t)(obj.cost(x,y,t,lambda));
        [theta_l, J, exit_flag] = fminunc(func,theta,options);
        
        obj.theta_l = theta_l;
        
    end
  
  end
  
end