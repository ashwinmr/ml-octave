function [theta, J_history] = Gradient_Descent(x,y,theta,alpha,num_iters,debugplot)
  % This function performs gradient descent on theta for a given x, y and number of iterations
  % Theta columns correspond to features of x
  
  if nargin < 6, debugplot = 0; end
  
  % Store useful values
  m = length(x);
  
  % Initialize
  J_history = zeros(num_iters,1);
  
  for i = 1:num_iters

    % Find the gradient
    gradient = x'*(x*theta-y)/m;
    
    % Take a gradient step
    theta = theta - alpha*gradient;
    
    % Store the cost in history
    J_history(i) = Cost_Function(x,y,theta);
    
  end
  
  % Plotting
  if debugplot
    
    plot(J_history);
    xlabel('Iteration');
    ylabel('Cost');
    grid on;
    title('Gradient Descent');
    
  end
  
end