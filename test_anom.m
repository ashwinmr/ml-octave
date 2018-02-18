load('ex8data1.mat');
x = X;
anom = anom_c();

anom.estimate(x);

% Get multivariate gaussian density at each x
p = anom.gaussian_multi(x);

% Select threshold using crossvalidation data
pval = anom.gaussian_multi(Xval);
epsilon = anom.select_threshold(pval,yval);

% Find outliers in training set
outliers = find(p < epsilon);
