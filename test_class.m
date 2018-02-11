% Testing all functions in the class

clc;
clear;
close;

addpath('Functions');
addpath('Classes');

% Load training data
load('p');
load('v');

% Create an object of linr class
linr = linr_c();

% Set the x and y for the class
x = [p,p.^0.5];
y = v;

% Perform gradient descent
alpha = 0.1;
max_iter = 1500;
lambda = 0;
linr.normal_solve(x,y);

% Predict
y_predicted = linr.predict(x);

% Plot
plot(p,v,'*'); % Plot the training data
hold all;
plot(x(:,1),y_predicted); % Plot the prediction
grid on;
xlabel('x');
ylabel('y');
legend('Training Data','Prediction');
hold off;