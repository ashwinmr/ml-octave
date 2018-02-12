classdef logr_c < handle

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
        % This function computes the cost function for logistic regression
        % It allows regularization
        % Theta is in the rolled form
        % Gradient is returned in unrolled form so it can be used by optimization algos
        if nargin < 5, lambda = 0; end

        % Set constants
        m = size(x,1);

        h = Sigmoid(x*theta');

        % Compute cost
        J = sum(-y.*log(h)-(1-y).*log(1-h))/m + sum(theta(:,2:end).^2)*lambda/2/m;

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
    
    function [pred_raw,pred] = predict(obj,x,theta_l,threshold)
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
        pred_raw = Sigmoid(x*theta_l');
        pred = pred_raw >= threshold;
      
    end
    
    function [err,ind,fscore] = pred_error(obj,x,y,theta_l,threshold)
        % This function calculates the error of prediction for an x and y
        if nargin < 4, theta_l = obj.theta_l; end
        if nargin < 5, threshold = 0.5; end
            
        % set constants
        m = size(x,1);
        
        [pred_raw,pred] = obj.predict(x,theta_l,threshold);
        ind = find(pred ~= y); % Indices of errors
        
        err = sum(-y.*log(pred_raw)-(1-y).*log(1-pred_raw))/m;
        
        % Calculating f_score
        stp = sum(sum(and(pred,y))); % True positive
        sfp = sum(sum(and(pred,~y))); % False positive
        sfn = sum(sum(and(~pred,y))); % False negative
        
        prec = stp/(stp + sfp); % Precision
        rec = stp/(stp + sfn); % Recall
        
        fscore = 2*(prec*rec)/(prec + rec);
        
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
