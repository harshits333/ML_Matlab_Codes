function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
m=length(X);
mu(1)=sum(X(:,1))/m;
mu(2)=sum(X(:,2))/m;

mat=zeros(m,2);

for i=1:m
  mat(i,1)=mu(1);
  mat(i,2)=mu(2);
     
   end;
   
   sigma(1)=sqrt((1/m)*sum((X(:,1)-mat(:,1)).^2));
   sigma(2)=sqrt((1/m)*sum((X(:,2)-mat(:,2)).^2));


for i=1:m

X(i,1)=(X(i,1)-mu(1))/sigma(1);
X(i,2)=(X(i,2)-mu(2))/sigma(2);

end;

X_norm=X;


% ============================================================

end
