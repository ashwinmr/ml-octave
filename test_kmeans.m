addpath('Classes');
addpath('Functions');

% Load an example dataset that we will be using
load('ex7data2.mat');

kmeans = kmeans_c();

k = 3;
initial_centroids = [3 3; 6 2; 8 5];

idx = kmeans.find_closest(X,initial_centroids);

centroids = kmeans.compute_centroids(X,idx,k);

centroids = kmeans.learn(X,k);