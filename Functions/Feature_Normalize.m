function [x,mu,sigma] = Feature_Normalize(x,mu,sigma)
% This function normalizes all the features of x using the mean and standard deviation of each feature
    if nargin < 2
        mu = mean(x);
    end

    if nargin < 3
        sigma = std(x);
    end

  % This function normalizes all the features of x using an input mean and standard deviation
  temp = bsxfun(@minus,x,mu);
  x = bsxfun(@rdivide,temp,sigma);
end
