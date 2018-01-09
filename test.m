% Prototype neural network code

nu; % Array of number of units in each layer from input to output
nl = length(nu); % Number of layers
m = 10; % Number of examples

% Input 
x = ones(1,10);

% Calculate cost for neural network

% Iterate for each layer
a{1} = x; % The first layer activation is x itself
for i = 1:nl-1
    
    % Store values
    act = a{i}; % Activation for current layer
    t = theta{i}; % Get the theta from current layer to next layer
    
    % Calculate values
    z = act*t';
    a{i+1} = Add_Bias(Sigmoid(z));
end

h = a{end};

% Get cost function by adding up cost for every single element
J = sum(sum(-log(h).*y-log(1-h).*(1-y)))/m;

% Adding Regularization
for i = 1:nl-1
    J = J + lambda*(sum(sum(theta{i}(:,2:end).^2));
end

% Back propogation






