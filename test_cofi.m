addpath('Classes');
%  Load data
load ('ex8_movies.mat');

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
Y(1,3) = 2; % Add my own ratings
R(1,3) = 1;
R = R(1:num_movies, 1:num_users);

cofi = cofi_c();
p = [X(:);Theta(:)];
cost = cofi.cost(p,num_features,Y,R,1.5);

cofi.learn(Y,R,10,1.5,X,Theta);

pred = cofi.predict();