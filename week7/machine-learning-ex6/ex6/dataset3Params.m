function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% =========================================================================

params = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
m = size(params, 1);
%C_test = 0.3;

for  i = 1:m,
  C_test = params(i);
  for j = 1:m,
    sigma_test = params(j);
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    train_error = mean(double(predictions ~= yval));

    if ~exist('min_error', 'var') || isempty(min_error),
      min_error = train_error;
    end;
    
    if train_error < min_error,
      min_error = train_error;
      C = C_test;
      sigma = sigma_test;
    end;
  end;
end;


% =========================================================================

end
