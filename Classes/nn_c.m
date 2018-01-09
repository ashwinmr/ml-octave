classdef nn_c < handle

  properties
    
    nl; % Number of layers
    xtr; % Training examples
    ytr;
    theta; % Cell array of matrices
    
  end
  
  methods
  
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
  
    function cost(obj,x,y,theta,lambda)
        
        nl = obj.nl;
        a{1} = x; % First layer activation is the examples itself
        
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
        
        
    end
  
  end