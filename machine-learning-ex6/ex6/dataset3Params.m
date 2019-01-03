function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err_min = 1;

for i=1:8

  for n=1:8

    C_2 = values(i);
    sigma_2 = values(n);
    fprintf('Testing :\nC : %f\nsigma : %f\n', C_2, sigma_2)
    % run the function
    model = svmTrain(X, y, C_2, @(x1, x2) gaussianKernel(x1, x2, sigma_2));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    if err < err_min
      fprintf('New min found at : \nError: %f\nC : %f\nsigma : %f\n', err_min, C, sigma)
      err_min = err;
      C = C_2;
      sigma = sigma_2;
    end
    
    n = n + 1;
  end

  i = i + 1;
end





% =========================================================================

end
