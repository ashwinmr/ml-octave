function J = Cost_Function(x,y,theta)
  % This function computes the cost function for a given x, y and theta.
  % The cost function used is for linear regression
  % x is the training examples (rows) having features (cols) and a bias unit
  % y is the training result
  % theta is the coefficients used as a linear function of x (including bias unit)
  
  % Store values
  m = length(x);
  
  % Calculate cost function
  J = (x*theta-y)'*(x*theta-y)/2/m;
  
end
