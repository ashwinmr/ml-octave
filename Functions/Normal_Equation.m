function [theta] = Normal_Equation(x,y)
  % This function obtains the optimal theta for linear regression without gradient descent
  % Note that the theta calculated using the normal equation will be different from
  % the one you get using gradient descent since the normal equation does not use
  % feature normalization
  
  theta = (x'*x)^-1*x'*y;

end
