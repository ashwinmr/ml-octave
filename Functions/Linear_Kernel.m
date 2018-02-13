function sim = Linear_Kernel(x1, x2)
    % This function gives the similarity between two inputs using linear kernel
    % Make sure that x1 and x2 are normalized before sending them to this function,
    % Otherwise, the similarity will be heavily weighted to some of their features

    sim = x1*x2'; % dot product
end