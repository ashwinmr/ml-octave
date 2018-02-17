% Load data
load ('ex7data1.mat');
x = X;
pca = pca_c();

% Run pca
pca.init(x);

% Reduce dimensions to 1
z = pca.project(x,1);

% Recover data
x_rec = pca.recover(z);

% Check variance 
var = pca.check(1);

% Plotting
plot(x(:,1),x(:,2),'*');
hold all;
plot(x_rec(:,1),x_rec(:,2),'*');
legend('original','recovered');

