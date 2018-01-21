function x = Add_Bias(x)
  % This function adds the bias unit to an input training set
  x = [ones(size(x,1),1),x];
end