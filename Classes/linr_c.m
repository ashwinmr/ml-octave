classdef linr_c < handle

  properties
  
    theta_l; % Learned theta
    mu;
    sigma;
    
  end
  
  methods
  
    function [] = init(obj,x,y)
        
        % Feature Normalize
        [~,mu,sigma] = Feature_Normalize(x);
        obj.mu = mu;
        obj.sigma = sigma;
        
        % Initialize theta
        nf = size(x,2);
        nc = size(y,2);
        obj.theta_l = zeros(nc,nf+1);
        
    end
    
    function [J, gradient] = cost(obj,x,y,theta,lambda)
      % This function computes the cost function for x, y and theta.
      % It allows regularization
      % theta is a row vector corresponding to x
      % Gradient is returned in unrolled form so it can be used by optimization algos
      % x should already contain bias term
      if nargin < 5, lambda = 0; end
      
      % set constants
      m = size(x,1);
      
      h = x*theta';
      
      % Calculate cost function
      J = sum((h-y).^2)/2/m + sum(theta(:,2:end).^2)*lambda/2/m;
      
      % Compute gradient
      gradient = (x'*(h-y)/m)' + [zeros(size(theta,1),1),theta(:,2:end)*lambda/m];
      gradient = Unroll(gradient);

    end
    
    function [theta_l,J_history] = learn_grad(obj,x,y,alpha,num_iters,lambda,theta)
    % This function uses machine learning to learn a model for training data
    % x and y using gradient descent
        if nargin < 5, num_iters = 500; end
        if nargin < 6, lambda = 0; end
        if nargin < 7, theta = obj.theta_l; end

        % set constants
        m = size(x,1);
        nf = size(x,2);
        nc = size(y,2);
        mu = obj.mu;
        sigma = obj.sigma;
        
        % Feature Normalize
        x = Feature_Normalize(x,mu,sigma);

        % Add bias
        x = [ones(m,1),x];
        
        % Initialize J_history
        J_history = zeros(num_iters,1);
        
        for i = 1:num_iters

            [J_history(i),gradient] = obj.cost(x,y,theta,lambda);
            
            % Roll the gradient
            gradient = Roll_Theta(gradient,[nf;nc]);

            % Take a gradient step
            theta = theta - alpha*gradient;

        end
        
        theta_l = theta;
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
    
    function [err] = pred_error(obj,x,y,theta_l)
        % This function calculates the error of prediction for an x and y
        if nargin < 4, theta_l = obj.theta_l; end
        
        % set constants
        m = size(x,1);
        
        pred = obj.predict(x,theta_l);
        
        err = sum((pred - y).^2)/2/m;
        
    end
    
    function [theta_l] = learn_normal(obj,x,y)
        % This function obtains the optimal theta for linear regression without gradient descent
        % Note that the theta calculated using the normal equation will be different from
        % the one you get using gradient descent since the normal equation does not use
        % feature normalization

        % set constants
        m = size(x,1);
        
        % Add bias
        x = [ones(m,1),x];
        
        % Use normal equation to solve
        theta_l = (x'*x)^-1*x'*y;
        theta_l = theta_l';
        obj.mu = 0;
        obj.sigma = 1;

        obj.theta_l = theta_l;

    end
    
    function [theta_l,J] = learn(obj,x,y,max_iter,lambda,theta)
        % This function finds the optimal theta for a training set using fminunc
        if nargin < 4; max_iter = 500; end
        if nargin < 5; lambda = 0; end
        if nargin < 6; theta = obj.theta_l; end

        % set constants
        m = size(x,1);
        nf = size(x,2);
        nc = size(y,2);
        mu = obj.mu;
        sigma = obj.sigma;
        
        % Feature Normalize
        x = Feature_Normalize(x,mu,sigma);

        % Add bias
        x = [ones(m,1),x];
            
        % Set options
        options = optimset('GradObj', 'on', 'MaxIter', max_iter);
        
        % Optimize
        func = @(t)(obj.cost(x,y,Roll_Theta(t,[nf,nc]),lambda));
        [theta_l, J, exit_flag] = fminunc(func,Unroll(theta),options);
        
        % Roll theta_l
        theta_l = Roll_Theta(theta_l,[nf;nc]);
        obj.theta_l = theta_l;
        
    end

  end
  
end