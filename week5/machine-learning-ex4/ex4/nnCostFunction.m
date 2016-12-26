function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% creating y_matrix in the form of binary tags
y_matrix = eye(num_labels)(y,:);

%================== FORWARD PROPAGATION ======================

% X has size 5000 x 400
a1 = [ones(m,1) X];
% a1 has size 5000 x 401
% Theta1 has size 25 x 401 
% z2 and a2 has size 5000 x 25
z2 = a1 * Theta1';
a2 = sigmoid(z2);
% adding the bias column 
% a2 has size 5000 x 26
a2 = [ones(rows(a2),1) a2];
% Theta2 has size 10 x 26
% z3 and a3 has size 5000 x 10
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% ___________________________________________________________

% ============== COMPUTING THE COST FUNCTION ================

% ===== The Regularization Term ======
Theta1_reg = Theta1; %[zeros(rows(Theta1),1) Theta1(:,2:end)];
Theta1_reg(:,1) = 0;
Theta2_reg = Theta2; %[zeros(rows(Theta2),1) Theta2(:,2:end)];
Theta2_reg(:,1) = 0;
sumTheta1 = sum(sum((Theta1_reg .^ 2), 2));
sumTheta2 = sum(sum((Theta2_reg .^ 2), 2));
reg = lambda * (sumTheta1 + sumTheta2) / (2 * m);
%___________________________________________________________

% ===== Apply regularization to the cost function =====
costs = (y_matrix .* log(a3)) + ((1 - y_matrix) .* log(1 - a3));
sumCosts = sum(sum(costs, 2));
J = (-1/m) * sumCosts + reg;
% ___________________________________________________________


% ========== Backpropagation ============

% size 5000 x 10
d3 = a3 - y_matrix;
% size 5000 x 25
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

% Computing Deltas
Delta1 = d2' * a1;
Delta2 = d3' * a2;

% ========= Regularization of the gradient =============

Theta1_grad = (Delta1 / m) + ((Theta1_reg * lambda) / m);
Theta2_grad = (Delta2 / m) + ((Theta2_reg * lambda) / m);

% ___________________________________________________________

% =============== Unroll gradients ==========================
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% ___________________________________________________________

end
