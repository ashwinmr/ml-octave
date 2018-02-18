classdef anom_c < handle

    properties
        mu;
        sigma2;
    end
    
    methods
    
        function [mu,sigma2] = estimate(obj,x)
            
            mu = mean(x);
            sigma2 = var(x);
            
            obj.mu = mu;
            obj.sigma2 = sigma2;
            
        end
        
        function p = gaussian_multi(obj,x, mu, sigma2)
        %Computes the probability density function of the
        %multivariate gaussian distribution at each x
        
            if nargin < 3, mu = obj.mu; end
            if nargin < 4, sigma2 = obj.sigma2; end
            
            % Set constants
            k = size(x,2);
            
            if (size(sigma2, 2) == 1) || (size(sigma2, 1) == 1)
                sigma2 = diag(sigma2);
            end

            x = bsxfun(@minus, x, mu);
            p = (2 * pi) ^ (- k / 2) * det(sigma2) ^ (-0.5) * ...
            exp(-0.5 * sum(bsxfun(@times, x * pinv(sigma2), x), 2));

        end
        
        function [epsilon_best f1score_best] = select_threshold(obj,pval,yval)
        % Find the best threshold (epsilon) to use for selecting outliers
        % using a validation p set and ground truth y

        f1score_best = 0;
        f1score = 0;
        
        % Set constants
        stepsize = (max(pval) - min(pval)) / 1000;
        
            for epsilon = min(pval):stepsize:max(pval)

                predictions = pval < epsilon;
        
                tp = sum((predictions == 1) & (yval == 1));
                fp = sum((predictions == 1) & (yval == 0));
                fn = sum((predictions == 0) & (yval == 1));
        
                prec = tp/(tp + fp); % Precision
                rec = tp/(tp + fn); % Recall
        
                f1score = 2*prec*rec/(prec + rec);

                if f1score > f1score_best
                   f1score_best = f1score;
                   epsilon_best = epsilon;
                end
            end

        end
    
        function check_gaussian(obj,x)
            % This function plots histograms of all features of x
            % to let you judge if they are gaussian and transform them if needed
            
            % Set constants
            n = size(x,2);
            
            r = floor(sqrt(n));
            c = ceil(sqrt(n));
            
            for i = 1:n
                subplot(r,c,i);
                hist(x(:,i));
                xlabel(num2str(i));
                grid on;
            end
        end
        
        
    
    end
    
end