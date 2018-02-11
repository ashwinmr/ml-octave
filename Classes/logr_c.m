classdef logr_c < handle

  properties
  
    theta_l; % Learned theta
    mu;
    sigma;
    
  end
  
  methods
    
    function [J, gradient] = cost(obj,x,y,theta,lambda)
        % This function computes the cost function for logistic regression
        % It allows regularization
        if nargin < 5, lambda = 0; end

        % Set constants
        m = size(x,1);

        h = Sigmoid(x*theta');

        % Compute cost
        J = sum(-y.*log(h)-(1-y).*log(1-h))/m + sum(theta(:,2:end).^2)*lambda/2/m;

        % Compute gradient
        gradient = (x'*(h-y)/m)' + [zeros(size(theta,1),1),theta(:,2:end)*lambda/m];

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
    
    function [pred] = predict(obj,x,theta_l,threshold)
        % Function to predict the result of logistic regression
        if nargin < 3, theta_l = obj.theta_l; end
        if nargin < 4, threshold = 0.5; end
            
        % set constants
        m = size(x,1);
        mu = obj.mu;
        sigma = obj.sigma;
        
        % Feature Normalize
        x = Feature_Normalize(x,mu,sigma);
        
        % Add bias
        x = [ones(m,1),x];
        
        % Predict
        pred = Sigmoid(x*theta_l') >= threshold;
      
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