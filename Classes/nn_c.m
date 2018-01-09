classdef nn_c < handle

  properties
    
    nl; % Number of layers
    xtr; % Training examples
    ytr;
    theta; % Cell array of matrices
    
  end
  
  methods
  
    function t = unroll_theta(obj,theta)
        
        nl = obj.nl; % number of layers
        t = []; % Initialize t
        
        for i= 1:nl-1
            t = [t;theta{i}(:)];
        end
        
    end
    
    function theta = roll_theta(obj,t)
        
        nl = obj.nl; % number of layers
        nu = obj.nu; % number of units in each layer
        
        theta = cell(nl,1);
        count = 1;
        
        for i = 1:nl-1
            r = nu(i+1); % number of rows
            c = nu(i)+1; % number of columns
            theta{i} = reshape(t(count:r*c),r,c);
            count = count + r*c; % update the location of counter in the array
        end
        
    end
            
  
    function init_theta(obj,epsilon)
        
        if nargin < 2, epsilon = 0.12; end
        
        nl = obj.nl; % number of layers
        nu = obj.nu; % array of number of units in each layer
        
        theta = cell(nl,1);
        
        for i = 1:nl-1
            
            theta{i} = rand(nu(i+1),nu(i) + 1)*2*epsilon - epsilon;
            
        end
        
        % Set the object property
        obj.theta = theta;
        
    end
  
    function [J,theta_grad] = cost(obj,x,y,theta,lambda)
        
        nl = obj.nl;
        a{1} = x; % First layer activation is the examples itself
        theta_grad = cell(nl,1);
        m =size(y,1); 
        
        % Forward propagation
        for i = 1:nl-1
            
            act = a{i}; % Activation for current layer
            t = theta{i}; % Get the theta from current layer to next layer
            
            
            z = act*t';
            a{i+1} = Add_Bias(Sigmoid(z));
            
        end
        
        h = a{end};
        
        % Get cost function by adding up cost for every single element
        J = sum(sum(-log(h).*y-log(1-h).*(1-y)))/m;

        % Adding Regularization
        for i = 1:nl-1
            J = J + lambda*(sum(sum(theta{i}(:,2:end).^2));
        end
        
        % Back propogation
        delta = cell(nl,1);
        
        delta{end} = a{end} - y;
        
        for i = nl-1:-1:2
            t = theta{i};
            
            delta{i} = delta{i+1}*t(:,2:end).*Sigmoid_Gradient(z{i});
            
        end
        
        % Accumulate the gradients
        % You cannot ignore the bias units when you are calculating the total contribution to error
        for i = 1:nl-1
            theta_grad{i} = delta{i}'*a{i}/m;
            
            % Add regularization (ignore bias column)
            theta_grad{i}(:,2:end) = theta_grad(:,2:end) + theta{i}(:,2:end)*lambda/m;
            
        end
        
    end
  
  end