classdef ML_C < handle

  properties
    
    x; % Stored training set with bias and normalization
    y; % result of training set
    theta; % parameters for machine learning
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
    
    function g = sigmoid(z)
      % This function runs the sigmoid function on every element in a matrix

      g = 1./(1+exp(-z));

    end
    
  end
  
  methods
  
    % setters
    
    function Set_x(obj,x)
      % This function sets the value of x after feature normalization and biasing
      
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
    
    function [J, gradient] = Compute_Cost(obj)
      % This function computes the cost function for x, y and theta.
      % The cost function used is for linear regression
      % x is the training examples (rows) having features (cols) and a bias unit
      % y is the training result
      % theta is the coefficients used as a linear function of x (including bias unit)
      
      % local values
      m = length(obj.x);
      
      % Calculate cost function
      J = (obj.x*obj.theta-obj.y)'*(obj.x*obj.theta-obj.y)/2/m;
      
      h = obj.x*obj.theta;
      
      % Compute gradient
      gradient = obj.x'*(h-obj.y)/m;
  
    end
    
    function [J, gradient] = Compute_Cost_Logistic(obj,lambda)
      % This function computes the cost function for logistic regression

      % Store values
      m = length(obj.y); % number of training examples

      % You need to return the following variables correctly 
      J = 0;
      grad = zeros(size(theta));
      
      h = sigmoid(obj.x*obj.theta);
      
      % Compute cost
      J = (-obj.y'*log(h)-(1-obj.y)'*log(1-h))/m + (obj.theta(2:end)'*obj.theta(2:end))*lambda/2/m;
      
      % Compute gradient
      gradient = (obj.x'*(h-obj.y))/m + [0;obj.theta(2:end)*lambda/m];
      
    end
    
    function Gradient_Descent(obj,debugplot,alpha,num_iters)
      % This function performs gradient descent on theta for a given x, y and number of iterations
      
      if nargin < 2, debugplot = 0; end
      
      % Store useful values
      m = length(obj.y);
      
      % Initialize
      obj.J_history = zeros(num_iters,1);
      
      for i = 1:num_iters
        
        [obj.J_history(i),gradient] = obj.Compute_Cost();
        
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
    
    function Normal_Equation(obj)
      % This function obtains the optimal theta for linear regression without gradient descent
      % Note that the theta calculated using the normal equation will be different from
      % the one you get using gradient descent since the normal equation does not use
      % feature normalization
      
      obj.theta = (obj.x'*obj.x)^-1*obj.x'*obj.y;

    end
    
    function y = Predict(obj,x_in)
        
      % First feature normalize using stored values
      x = ML_C.Feature_Normalize(x_in,obj.mu,obj.sigma);
      
      % Add the bias unit
      x = ML_C.Add_Bias(x);
      
      % Get y
      y = x*obj.theta;
      
    end
    
  end
  
end