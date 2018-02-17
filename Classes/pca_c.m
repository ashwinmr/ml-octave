classdef pca_c < handle

    properties
        u;
        s;
        mu;
        sigma;
    end
    
    methods
    
        function [u,s] = init(obj,x)
            
            % Set constants
            m = size(x,1);
        
            % Feature Normalize
            [x,mu,sigma] = Feature_Normalize(x);
            obj.mu = mu;
            obj.sigma = sigma;
            
            % Run PCA only on training set
            Sig = x'*x/m;

            [u, s, v] = svd(Sig);
            
            obj.u = u;
            obj.s = diag(s);
        
        end
        
        function z = project(obj,x,k,u)
        % x has to be in the normalized form
        % You can project any data using the pca obtained from the training set
            if nargin < 4, u = obj.u; end
        
            % set constants
            m = size(x,1);
            mu = obj.mu;
            sigma = obj.sigma;
        
            % Feature normalize
            x = Feature_Normalize(x,mu,sigma);
   
            % Take the first k eigen vectors
            u_reduce = u(:,1:k);

            % Each example is reduced to a lower number of dimensions each obtained by multiplying by an eigen vector
            z = x*u_reduce;

        end
        
        function x_rec = recover(obj,z,u)
        %RECOVERDATA Recovers an approximation of the original data when using the 
        %projected data
            if nargin < 3, u = obj.u; end
        
            % Set constants
            k = size(z,2);
            mu = obj.mu;
            sigma = obj.sigma;
            
            u_reduce = u(:,1:k);
            x_rec = z*u_reduce';
            
            % Remove normalization
            x_rec = Feature_De_Normalize(x_rec,mu,sigma);

        end
        
        function var = check(obj,k)
        % Use this function to check the variance for different k
        % Try to keep variance above 85%
        % 99% of varience is retained with var = 0.99
            
            % Set constants
            n = length(k);
            s = obj.s;
            
            var = zeros(n,1);
            for i = 1:n
                var(i) = sum(s(1:k(i)))/sum(s);
            end
            
        end

    
    end
    
end