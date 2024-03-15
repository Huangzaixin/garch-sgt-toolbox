function [LL, likelihoods] = sgtdis_LL(theta, x)
% Purpose:
%       return the log-likelihood at x of SGT distribution
%       refer to Theodossiou (1998)
% Inputs:
%       theta = [lam; k; n]; 
%
% Output:
%       LL - the log likelikelihood
%       Likelihoods - the individual log likelihood
% Henry Mo, June 2005

lam = theta(1);
k   = theta(2);
n   = theta(3);

beta1=beta(n/k,1/k);
beta2=beta((n-1)/k,2/k);
beta3=beta((n-2)/k,3/k);
nk=(n+1)/k;
rho=2*lam/beta1*nk^(1/k)*beta2;
g=(1+3*lam^2)/beta1*nk^(2/k)*beta3;
theta=1/sqrt(g-rho^2);
delta=rho*theta;
C=0.5*k*nk^(-1/k)/beta1/theta;
z=x+delta;
absz=abs(z);
likelihoods=log(C*(1+(absz./(1+sign(z)*lam)/theta).^k/nk).^(-nk));
LL=-sum(likelihoods);

