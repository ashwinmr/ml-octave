function [x_norm, mu, sigma] = Feature_Normalize(x)
  % This function normalizes all the features of x using the mean and standard deviation of each feature
  % It is important to store the values of mu and sigma, since later, when using
  % New inputs, they have to be normalized using these values
  % Note that you have to exclude the bias input during feature normalization !
  
  mu = mean(x);
  sigma = std(x);
  x_norm = (x - mu)./sigma;
  
  % Check for bias unit
  if x(1,1) == 1
    disp('Bias unit was present in x. It was ignored');
    % Replace the NAN column in x_norm with ones
    x_norm(:,1) = 1;
  end
  
end
