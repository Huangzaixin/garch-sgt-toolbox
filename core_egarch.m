function [stdresid,error, mu, h] = core_egarch(data,parameters,flag_a0,ar,ma,flag_omega,p,q,o)
T = length(data);
m = max([ar,ma,p,q]);
arma_order_sum = ar + ma;  
subconst = sqrt(2/pi);

mu = zeros(T,1);                      % conditional mean
error = zeros(T,1);                   % error term
h = zeros(T,1);                       % conditional variance
stdresid = zeros(T,1); 

mu(1:m,1) = mean(data);               % Preallocate conditional mean
error(1:m,1) = data(1:m) - mu(1:m);   % Preallocate error
h(1:m,1) = var(data);                 % h(t) is conditional variance
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
        v = parameters(flag_a0+arma_order_sum+1); 
    else
        v = 0;     
    end
    
    for j=1:p
        v = v + parameters(flag_a0+arma_order_sum+flag_omega+j) * ( abs(stdresid(t-j,1)) - subconst );
    end
    for j=1:q
        v = v + parameters(flag_a0+arma_order_sum+flag_omega+p+j) * log(h(t-j,1));
    end
    for j=1:o
        v = v + parameters(flag_a0+arma_order_sum+flag_omega+p+q+j) * (stdresid(t-j,1));
    end
    
    h(t,1) = exp(v);
    stdresid(t,1) = error(t,1)/sqrt(h(t,1));
end

% stdresid = stdresid((m + 1):T,1);
% error = error((m + 1):T,1);
% h = h((m + 1):T,1);
% mu = mu((m + 1):T,1);
end
