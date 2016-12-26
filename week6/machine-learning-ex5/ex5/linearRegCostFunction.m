function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% ====================== COST ==========================

h = X*theta;
error_sqr = (h-y) .^ 2;
theta_reg = theta;
theta_reg(1,:) = 0;
theta_reg_sqr = theta_reg .^ 2;
J = (sum(error_sqr)/(2*m)) + (lambda*sum(theta_reg_sqr)/(2*m));

% ======================== GRADIENT ====================

grad = (X'*(h-y) + (lambda*theta_reg))/m;

% =========================================================================

grad = grad(:);

end
