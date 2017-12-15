function x = Add_Bias(x)
  % This function adds the bias unit to an array of input examples x
  
  x = [ones(size(x,1),1),x];
  
end