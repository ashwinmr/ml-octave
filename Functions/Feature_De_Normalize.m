function [x] = Feature_De_Normalize(x,mu,sigma)
    % This function removes the normalization applied to x using an
    % input mu and sigma
    temp = bsxfun(@times,x,sigma);
    x = bsxfun(@plus,temp,mu);
end