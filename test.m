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
    

