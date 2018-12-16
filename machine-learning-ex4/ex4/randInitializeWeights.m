function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%


%mx = 0.12^2;
%mn = -1 * mx;
mx = 0.12;
mn = -1 * mx;

fprintf(['\nSize of L_out & L_in \n'])
size(L_out)
size(L_in)

for i=1:L_in
  nums = mn + (mx-mn)*rand(1,L_out);
  nums = nums';
  fprintf(['/ / / / / / / / / / / / / / / / / / / / / / / / / / / /'])
  fprintf(['\nThese are the params : %f \n'], size(nums))
  fprintf(['/ / / / / / / / / / / / / / / / / / / / / / / / / / / /'])
  W(:,i) = nums;
end

%epsilon = 0.12;
%W = rand(L_out, 1 + L_in) * 2 * epsilon − epsilon ;




% =========================================================================

end
