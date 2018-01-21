function theta = Roll_Theta(t,nu)
% Roll up a vector of theta values into a matrix if theres only 2 layers and 
% into a cell array of matrices if there are more than 2 layers
% The minimum is 2 layers (input, output);

    nl = size(nu,1); % Number of layers
    
    if nl>2
        % There are multiple thetas, use a cell array
        theta = cell(nl-1,1);
        count = 1;
        
        for i = 1:nl-1
            r = nu(i+1); % number of rows
            c = nu(i)+1; % number of columns
            theta{i} = reshape(t(count:count-1+r*c),r,c);
            count = count + r*c; % update the location of counter in the array
        end
    else
        % Theres only 1 theta use a matrix
        r = nu(2); % number of rows
        c = nu(1)+1; % number of columns
        theta = reshape(t,r,c);
    end
    
end