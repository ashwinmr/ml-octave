function numgrad = Num_Grad(J,theta,epsilon)
% Computes the numerical gradient given a theta vector and a cost function pointer
% which uses theta as the only input
% J(theta) should give the cost at theta              

    if nargin < 3, epsilon = 1e-4;

    numgrad = zeros(size(theta));
    perturb = zeros(size(theta));

    for p = 1:numel(theta)
        % Set perturbation vector
        perturb(p) = epsilon;
        
        % Compute Numerical Gradient
        numgrad(p) = (J(theta + perturb) - J(theta - perturb))/(2*epsilon);
        perturb(p) = 0;
    end

end