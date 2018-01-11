function t = Unroll_Theta(theta)
% Unroll a theta matrix or cell array of matrices into a single vector

    t = []; % Initialize t

    % Check whether theta is a cell array
    if iscell(theta)
        nt = size(theta,1);
        % Loop for each theta
        for i= 1:nt
            t = [t;theta{i}(:)];
        end
    else
        % Theta is a matrix  
        t = theta(:);
    end
end