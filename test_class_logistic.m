% Testing all functions in the class

clc;
clear all;
close all;

addpath('Functions');
addpath('Classes');

% Load training data
load('x_log');
load('y_log');

% Create an object of ML class
ml = ML_C();

% Set the x and y for the class
ml.Set_x(x);
ml.Set_y(y);

% Perform logistic regression
lambda = 0;
n = 500;
ml.Optimize_Logistic(lambda,n);

% Predict
y_predicted = ml.Predict_Logistic(x);

% Plotting

% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,1)),  max(x(:,1))];
plot_x_n = ML_C.Feature_Normalize(plot_x,ml.mu,ml.sigma);
% Calculate the decision boundary line
plot_y_n = (-1./ml.theta(3)).*(ml.theta(2).*plot_x_n + ml.theta(1));
plot_y = ML_C.Feature_De_Normalize(plot_y_n,ml.mu,ml.sigma);

% Find indices of positive and negative examples
pos = find(y==1); 
neg = find(y==0);
plot(x(pos,1),x(pos,2),'k+','LineWidth',2,'MarkerSize',7);
hold all;
plot(x(neg,1),x(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);
plot(plot_x,plot_y);
grid on;
xlabel('x1');
ylabel('x2');
legend('Positive','Negative','boundary');
hold off;