function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% computing J 

prediction1 = sigmoid(X*theta);
prediction2 = 1 - prediction1;
temp = y .* log(prediction1) + (1-y) .* log( prediction2);
J = -1/m * sum(temp) + lambda/(2*m) * (theta(2:end)'*theta(2:end));

% computing grad
% separate solve
n = size(theta);
newsum = prediction1-y;
grad(1) = 1/m * ( newsum' * X(:,1));
grad(2:n) = 1/m * ((newsum'* X(:,2:n)))' + lambda/m * theta(2:n);





% =============================================================

end
