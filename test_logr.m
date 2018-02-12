% Testing all functions in the class

clc;
clear all;
close all;

addpath('Functions');
addpath('Classes');

% Load training data
load('x_log');
load('y_log');

% Create an object of logr class class
logr = logr_c();

% Perform logistic regression
alpha = 0.1;
lambda = 0;
max_iter = 500;
logr.init(x,y);
logr.learn_grad(x,y,alpha,max_iter,lambda);

% Predict
y_predicted = logr.predict(x);

% Plotting

% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,1)),  max(x(:,1))];
plot_x_n = Feature_Normalize(plot_x,logr.mu,logr.sigma);
% Calculate the decision boundary line
plot_y_n = (-1./logr.theta_l(3)).*(logr.theta_l(2).*plot_x_n + logr.theta_l(1));
plot_y = Feature_De_Normalize(plot_y_n,logr.mu,logr.sigma);

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