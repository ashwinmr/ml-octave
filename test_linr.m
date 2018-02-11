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

% Set constants
alpha = 0.1;
max_iter = 100;
lambda = 0;

% Learn
linr.learn(x,y,max_iter,lambda);
y_predicted = linr.predict(x);

%{
% Learning curves
ex = randperm(length(x));
m = [5:50];
Jt = zeros(length(m),1);
Jc = zeros(length(m),1);

for i = 1:length(m)
    
    % Training error
    Xt = x(ex(1:m(i)),:);
    Yt = y(ex(1:m(i)),:);
    linr.learn(Xt,Yt,max_iter,lambda);
    Jt(i) = linr.pred_error(Xt,Yt);
    
    % Cross validation error
    Xc = x(ex(300:end),:);
    Yc = y(ex(300:end),:);
    Jc(i) = linr.pred_error(Xc,Yc);
    
end

plot(m,Jt,m,Jc);
legend('J Train','J CV');
%}

% Plot
plot(p,v,'*'); % Plot the training data
hold all;
plot(x(:,1),y_predicted); % Plot the prediction
grid on;
xlabel('x');
ylabel('y');
legend('Training Data','Prediction');
hold off;
%}