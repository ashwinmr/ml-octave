classdef ML_C < handle

  properties
    
    x; % Stored training set with bias and normalization
    y; % result of training set (can have multiple columns)
    m; % Number of examples in the training set
    theta; % parameters for machine learning (if multiple, columns are theta for each class)
    mu = 0; % mean of x
    sigma = 1; % sigma of x
    J_history; % history of J
    
  end
  
  methods(Static)
  
    function x = Add_Bias(x)
      % This function adds the bias unit to an input training set
      x = [ones(size(x,1),1),x];
    end
    
    function x = Remove_Bias(x)
      % This function removes the bias unit from an input training set
      x = x(:,2:end);
    end
    
    function [x] = Feature_Normalize(x,mu,sigma)
      % This function normalizes all the features of x using an input mean and standard deviation
      temp = bsxfun(@minus,x,mu);
      x = bsxfun(@rdivide,temp,sigma);
    end
    
    function [x] = Feature_De_Normalize(x,mu,sigma)
        % This function removes the normalization applied to x using an
        % input mu and sigma
        temp = bsxfun(@times,x,sigma);
        x = bsxfun(@plus,temp,mu);
    end
    
    function g = Sigmoid(z)
      % This function runs the sigmoid function on every element in a matrix

      g = 1./(1+exp(-z));

    end
    
  end
  
  methods
  
    % setters
    
    function Set_x(obj,x)
      % This function sets the value of x after feature normalization and biasing
      
      % Store m
      obj.m = size(x,1);
      
      % Get the normalization values
      obj.mu = mean(x);
      obj.sigma = std(x);
      
      % Feature normalize x using these values
      x = ML_C.Feature_Normalize(x,obj.mu,obj.sigma);
      
      % Add bias unit to x
      x = ML_C.Add_Bias(x);
      
      % Store x
      obj.x = x;
      
      % Initialize theta
      obj.theta = zeros(size(obj.x,2),1);

    end
    
    function Set_y(obj,y)
        % This function sets the value of y
        obj.y = y;
    end
    
    function x = Get_x(obj)
        % This function gets the value of x be removing the bias and
        % normalization
        
        % Remove bias unit
        x = ML_C.Remove_Bias(obj.x);
        
        % Remove normalization
        x = ML_C.Feature_De_Normalize(x,obj.mu,obj.sigma);
    end
    
    function [J, gradient] = Compute_Cost_Linear(obj,lambda,theta)
      % This function computes the cost function for x, y and theta.
      % It allows regularization
      if nargin < 2, lambda = 0; end
      if nargin < 3, theta = obj.theta; end
          
      % local values
      m = obj.m;
      
      % Calculate cost function
      J = sum((obj.x*theta-obj.y).^2)/2/m + sum(theta(2:end,:).^2)*lambda/2/m;
      
      h = obj.x*theta;
      
      % Compute gradient
      gradient = obj.x'*(h-obj.y)/m;
  
    end
    
    function [J, gradient] = Compute_Cost_Custom(obj,lambda,theta)
      % This function computes the cost function for x, y and theta.
      % It allows regularization
      if nargin < 2, lambda = 0; end
      if nargin < 3, theta = obj.theta; end
          
      % local values
      m = length(obj.y);
      
      % Compute h of x
      c = theta(1);
      k = theta(2);
      b = theta(3);
      p = obj.x;
      p(p<0.1) = 0.1; % prevent zero for division
      v = (1-c./(p + c)).*(k.*(p+c)+b);
      h = v;
      
      % Remove NaNs
      h(isnan(h)) = 0;
      
      % Calculate cost function
      J = (h-obj.y)'*(h-obj.y)/2/m + (theta(2:end)'*theta(2:end))*lambda/2/m;
      
      % Compute gradients
      gradient(1,1) = sum((h-obj.y).*(k*(1-c./(p+c))+ (k*(p+c)+b).*(2*c-p)./((p+c).^2)))/m;
      gradient(2,1) = sum((h-obj.y).*p)/m;
      gradient(3,1) = sum((h-obj.y).*(1-c./(p+c)))/m;
      
    end
    
    function [J, gradient] = Compute_Cost_Logistic(obj,lambda,theta)
      % This function computes the cost function for logistic regression
      % It allows regularization
      if nargin < 2, lambda = 0; end
      if nargin < 3, theta = obj.theta; end

      % Store values
      m = obj.m;
      
      h = ML_C.Sigmoid(obj.x*theta);
      
      % Compute cost
      J = (-obj.y'*log(h)-(1-obj.y)'*log(1-h))/m + (theta(2:end)'*theta(2:end))*lambda/2/m;
      
      % Compute gradient
      gradient = (obj.x'*(h-obj.y))/m + [0;theta(2:end)*lambda/m];
      
    end
    
    function Gradient_Descent(obj,debugplot,alpha,num_iters,lambda)
      % This function performs gradient descent on theta for a given x, y and number of iterations
      
      if nargin < 2, debugplot = 0; end
      if nargin < 5, lambda = 0; end
      
      % Store useful values
      m = length(obj.y);
      
      % Initialize
      obj.J_history = zeros(num_iters,1);
      
      for i = 1:num_iters
        
        [obj.J_history(i),gradient] = obj.Compute_Cost_Linear(lambda);
        
        % Take a gradient step
        obj.theta = obj.theta - alpha*gradient;
        
      end
      
      % Plotting
      if debugplot
        
        plot(obj.J_history);
        xlabel('Iteration');
        ylabel('Cost');
        grid on;
        title('Gradient Descent');
        
      end
      
    end
    
    function Gradient_Descent_Custom(obj,debugplot,alpha,num_iters,lambda)
      % This function performs gradient descent on theta for a given x, y and number of iterations
      
      if nargin < 2, debugplot = 0; end
      if nargin < 5, lambda = 0; end
      
      % Store useful values
      m = length(obj.y);
      
      % Initialize
      obj.J_history = zeros(num_iters,1);
      
      for i = 1:num_iters
        
        [obj.J_history(i),gradient] = obj.Compute_Cost_Custom(lambda);
        
        % Take a gradient step
        obj.theta = obj.theta - alpha*gradient;
        
      end
      
      % Plotting
      if debugplot
        
        plot(obj.J_history);
        xlabel('Iteration');
        ylabel('Cost');
        grid on;
        title('Gradient Descent');
        
      end
      
    end
    
    function Optimize_Logistic(obj,lambda,max_iter)
        % This function finds the optimal theta for a training set using fminunc
        if nargin < 2; lambda = 0; end
        if nargin < 3; max_iter = 500; end
            
        % Set values
        theta = obj.theta;
            
        % Set options
        options = optimset('GradObj', 'on', 'MaxIter', max_iter);
        
        % Optimize
        func = @(t)(obj.Compute_Cost_Logistic(lambda,t));
        [theta, J, exit_flag] = fminunc(func,theta,options);
        
        obj.theta = theta;
        
    end
    
    function Optimize_Linear(obj,lambda,max_iter)
        % This function finds the optimal theta for a training set using fminunc
        if nargin < 2; lambda = 0; end
        if nargin < 3; max_iter = 500; end
            
        % Set values
        theta = obj.theta;
            
        % Set options
        options = optimset('GradObj', 'on', 'MaxIter', max_iter);
        
        % Optimize
        func = @(t)(obj.Compute_Cost_Linear(lambda,t));
        [theta, J, exit_flag] = fminunc(func,theta,options);
        
        obj.theta = theta;
        
    end
    
    function Optimize_Custom(obj,lambda,max_iter)
        % This function finds the optimal theta for a training set using fminunc
        if nargin < 2; lambda = 0; end
        if nargin < 3; max_iter = 500; end
            
        % Set values
        theta = obj.theta;
            
        % Set options
        options = optimset('GradObj', 'on', 'MaxIter', max_iter);
        
        % Optimize
        func = @(t)(obj.Compute_Cost_Custom(lambda,t));
        [theta, J, exit_flag] = fminunc(func,theta,options);
        
        obj.theta = theta;
        
    end

    function Normal_Equation(obj)
      % This function obtains the optimal theta for linear regression without gradient descent
      % Note that the theta calculated using the normal equation will be different from
      % the one you get using gradient descent since the normal equation does not use
      % feature normalization
      
      obj.theta = (obj.x'*obj.x)^-1*obj.x'*obj.y;

    end
    
    function y = Predict_Linear(obj,x_in)
        
      % First feature normalize using stored values
      x = ML_C.Feature_Normalize(x_in,obj.mu,obj.sigma);
      
      % Add the bias unit
      x = ML_C.Add_Bias(x);
      
      % Get y
      y = x*obj.theta;
      
    end
    
    function y = Predict_Logistic(obj,x_in,threshold)
        % Function to predict the result of logistic regression
        if nargin < 3, threshold = 0.5; end
        
      % First feature normalize using stored values
      x = ML_C.Feature_Normalize(x_in,obj.mu,obj.sigma);
      
      % Add the bias unit
      x = ML_C.Add_Bias(x);
      
      % Get y
      y = ML_C.Sigmoid(x*obj.theta) >= threshold;
      
    end
    
  end
  
end