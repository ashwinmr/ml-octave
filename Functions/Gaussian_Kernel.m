function sim = Gaussian_Kernel(x1, x2, sigma)
    % This function gives the similarity between two inputs using gaussian kernel
    % Make sure that x1 and x2 are normalized before sending them to this function,
    % Otherwise, the similarity will be heavily weighted to some of their features
    % Sigma here is a parameter for the gaussian kernel, do not confuse with std of training set.

    sim = exp(-(x1-x2)*(x1-x2)'/2/(sigma^2));
  
end