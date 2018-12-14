function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


size(Theta1)
size(Theta2)
size(X)

X = [ones(size(X, 1), 1) X];

size(X)

l2 = X * Theta1';


size(l2)

l2 = [ones(size(l2, 1), 1) sigmoid(l2)];
%l2 = [zeros(size(l2), 1) l2];

size(l2)

l3 = l2 * Theta2';

i = sigmoid(l3);

[m, i] = max(i, [], 2)
i(i==10)=0;
p = i




% =========================================================================


end
