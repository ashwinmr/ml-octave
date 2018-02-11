classdef nn_c < handle

  properties
    
    nu; % Array of number of units in each layer (exclude bias, include input and output layer)
    theta_l; % Learned theta
    
  end
  
  methods
  
    function pred = predict(obj,x,theta_l,threshold)
    % Predict the output for an input x and theta
    % If theta is not provided, the learned theta is used
    if nargin < 3, theta_l = obj.theta_l; end
    if nargin < 4, threshold = 0.5; end
        
        nu = obj.nu; % array of number of units in each layer
        nl = size(nu,1); % number of layers

        a = Add_Bias(x); % First layer activation is the examples itself. a has bias
        
        % Forward propagation
        for i = 1:nl-1
            z = a*theta_l{i}';
            a = Add_Bias(Sigmoid(z));
        end
        
        % Remove bias from the output
        a = Remove_Bias(a); % The output should not have bias
        
        pred = a >= threshold;
        
    end
    
    function [err] = pred_error(obj,x,y,theta_l)
        % This function calculates the error of prediction for an x and y
        
        # Cost function with lambda = 0 is the prediction error
        err = obj.cost(x,y,theta_l,0)
        
    end
  
    function [theta_l,J] = learn(obj,x,y,iters,lambda,algo)
    % Learns the theta using training set and selected algorithm
    % algorithm can be fminunc or fmincg
    
        if nargin < 6, algo = 'fmincg'; end
            
        nu = obj.nu; % array of number of units in each layer
        initial_theta = obj.init_theta(); % Initial theta
        
        % Create function pointer for the cost function to be minimized
        cost_func = @(t) obj.cost(x,y,Roll_Theta(t,nu),lambda);
        
        if strcmp(algo,'fminunc')
            % Use fminunc algorithm
            
            % Set options
            options = optimset('GradObj', 'on', 'MaxIter', iters);
            
            % Optimize
            [theta,J,exit_flag] = fminunc(cost_func,Unroll(initial_theta),options);
            
        else
            % Use the default fmincg algorithm
            
            % Set options
            options = optimset('MaxIter', iters);
            
            % Optimize
            [theta,J] = Fmincg(cost_func,Unroll(initial_theta),options);
            
        end
        
        % Reroll theta
        theta_l = Roll_Theta(theta,nu);
        
        % Store the learned theta
        obj.theta_l = theta_l;
        
    end
 
    function theta = init_theta(obj,epsilon)
    % Initialize theta using randomization within a small epsilon value
        if nargin < 2, epsilon = 0.12; end

        nu = obj.nu; % array of number of units in each layer
        nl = size(nu,1); % number of layers
        
        theta = cell(nl,1);
        
        for i = 1:nl-1
            
            theta{i} = rand(nu(i+1),nu(i) + 1)*2*epsilon - epsilon;
            
        end
        
    end
  
    function [J,theta_grad] = cost(obj,x,y,theta,lambda)
    % Computes the cost and gradient
    % The x input should not contain bias term
    % Theta should be in the rolled form
    % theta_grad output is in the unrolled form so it can be used by optimization algos
    
    
        if nargin < 5, lambda = 0; end
        
        nu = obj.nu; % array of number of units in each layer
        nl = size(nu,1); % number of layers
        a = cell(nl,1); % Initialize activations
        z = cell(nl,1); % Initialize z
        delta = cell(nl,1); % Initialize delta
        theta_grad = cell(nl,1);
        m =size(y,1); 
        
        z{1} = x; % This is not actually used. z does not contain bias
        a{1} = Add_Bias(x); % First layer activation is the examples itself. a has bias
        
        % Forward propagation
        for i = 1:nl-1
            
            z{i+1} = a{i}*theta{i}';
            a{i+1} = Add_Bias(Sigmoid(z{i+1}));
            
        end
        
        % Remove bias from the output
        a{end} = Remove_Bias(a{end}); % The output should not have bias
        h = a{end};
        
        % Get cost function by adding up cost for every single element
        J = sum(sum(-log(h).*y-log(1-h).*(1-y)))/m;

        % Adding Regularization
        for i = 1:nl-1
            J = J + lambda*(sum(sum(theta{i}(:,2:end).^2)))/2/m;
        end
        
        % Back propogation
        delta{end} = a{end} - y;
        
        for i = nl-1:-1:2
            t = theta{i};
            
            delta{i} = delta{i+1}*t(:,2:end).*Sigmoid_Gradient(z{i});
            
        end
        
        % Accumulate the gradients
        % You cannot ignore the bias units when you are calculating the total contribution to error
        for i = 1:nl-1
            theta_grad{i} = delta{i+1}'*a{i}/m;
            
            % Add regularization (ignore bias column)
            theta_grad{i}(:,2:end) = theta_grad{i}(:,2:end) + theta{i}(:,2:end)*lambda/m;
            
        end
        
        % Unroll the gradients
        theta_grad = Unroll(theta_grad);
        
    end
  
  end
  
end