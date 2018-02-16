classdef kmeans_c < handle

    properties
    
    end
    
    methods
    
        function centroids = init(obj,x,k)
        %KMEANSINITCENTROIDS This function initializes k centroids that are to be 
        %used in K-Means on the dataset x

            % Randomly reorder the indices of examples
            randidx = randperm(size(x, 1));
            % Take the first K examples as centroids
            centroids = x(randidx(1:k), :);
        end
        
        function idx = find_closest(obj,x,centroids)
        %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
        % Centroids is a matrix with each row representing a centroid

            % Initialize idx 
            idx = zeros(size(x,1), 1);

            m = size(x,1);

            % Loop through all examples
            for i = 1:m
                diff = x(i,:)-centroids;
                norm = sum(diff.^2,2);
                [~,idx(i)] = min(norm);
            end
        end
        
        function centroids = compute_centroids(obj,x,idx,k)
        %COMPUTECENTROIDS returns the new centroids by computing the means of the 
        %data points assigned to each centroid.

            % Loop through all centroids
            for i = 1:k
                points = x(idx == i,:);
                centroids(i,:) = mean(points);
            end

        end
        
        function [centroids,idx] = learn(obj,x,k,max_iters,initial_centroids)
        %RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
        %is a single example
        
        if nargin < 4, max_iters = 50; end
        if nargin < 5, initial_centroids = obj.init(x,k); end

            % Initialize values
            m = size(x,1);
            centroids = initial_centroids;
            previous_centroids = centroids;
            idx = zeros(m, 1);

            % Run K-Means
            for i=1:max_iters
                
                % For each example in x, assign it to the closest centroid
                idx = obj.find_closest(x, centroids);
                
                % Given the memberships, compute new centroids
                centroids = obj.compute_centroids(x,idx,k);
            end
        
        end

    end
    
end