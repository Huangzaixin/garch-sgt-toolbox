function [stdresid, error, mu, h] = core_basicgarch(data,parameters,flag_a0,ar,ma,flag_omega,p,q)
% PURPOSE: return standardized residual, error and conditional variance of Engle's GARCH
% INPUTS: 
%         data: return
%   parameters: the parameter set
%       ar, ma: orders of conditional mean equation
%         p, q: orders of conditional variance model  Default: p=1, q=1
% RETURNS:          
%     stdresid: the standardized residual of Engle's GARCH
%        error: error series of Engle's GARCH
%            h: conditional variance series
% AUTHOR: Zaixin Huang
% EMAIL:  zxhuang@mail.ccnu.edu.cn
% Download from https://github.com/Huangzaixin
% -------------------------------------------------------------------------
%%
T = size(data,1);
m = max([ar,ma,p,q]);

mu = zeros(T,1);         % conditional mean
error = zeros(T,1);      % error
h = zeros(T,1);          % conditional variance
stdresid = zeros(T,1);   % standardized residuals

mu(1:m,1) = mean(data);                 % initial conditional mean
error(1:m,1) = data(1:m,1) - mu(1:m,1);    
h(1:m,1) = var(data);                   % h(t) is conditional variance
stdresid(1:m,1) = error(1:m,1)./sqrt(h(1:m,1)); 

for t = (m + 1):T       
    if flag_a0 == 1
        mu(t,1) = parameters(flag_a0);
    end
    for j=1:ar
        mu(t,1) = mu(t,1) + parameters(flag_a0+j)*data(t-j,1);
    end
    for j=1:ma
        mu(t,1) = mu(t,1) + parameters(flag_a0+ar+j)*error(t-j,1);
    end   
    error(t,1) = data(t,1) - mu(t,1); 
    
    % include omega
    if isequal(flag_omega,1)
        h(t,1) = parameters(flag_a0+ar+ma+1); 
    else
        h(t,1) = 0;     
    end
    
    for j=1:p
        h(t,1) = h(t,1) + parameters(flag_a0+ar+ma+flag_omega+j)*(error(t-j,1))^2;
    end
    
    for j=1:q
        h(t,1) = h(t,1) + parameters(flag_a0+ar+ma+flag_omega+p+j)*h(t-j,1); 
    end   
    stdresid(t,1) = error(t,1)/sqrt(h(t,1)); 
end

% stdresid = stdresid((m + 1):T,1); 
% error = error((m + 1):T,1); 
% h = h((m + 1):T,1); 
% mu = mu((m + 1):T,1);
end
