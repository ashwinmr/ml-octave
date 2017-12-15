% Testing all functions in the class

clc;
clear;
close;

addpath('Functions');
addpath('Classes');

% Load training data
load('p');
load('v');

% Create an object of ML class
ml = ML_C();

% Set the x and y for the class
x = [p,p.^2,p.^3];
ml.Set_x(x);
ml.Set_y(v);

% Perform gradient descent
alpha = 0.1;
n = 1500;
ml.Gradient_Descent(1,alpha,n);

% Predict
y_predicted = ml.Predict(x);

% Plot
plot(p,v,'*'); % Plot the training data
hold all;
plot(x(:,1),y_predicted); % Plot the prediction
grid on;
xlabel('x');
ylabel('y');
legend('Training Data','Prediction');
hold off;