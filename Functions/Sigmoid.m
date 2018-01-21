function g = Sigmoid(z)
  % This function runs the sigmoid function on every element in a matrix

  g = 1./(1+exp(-z));

end