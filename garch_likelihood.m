function [LLF,stdresid,error,mu,h] = garch_likelihood(parameters,edata,flag_a0,ar,ma,flag_omega,p,q,o,m,garchtype,errortype)    
T = size(edata,1);

switch garchtype
    case 0   % Engle's Garch
        o = 0;
        [stdresid, error, mu, h] = core_basicgarch(edata,parameters,flag_a0,ar,ma,flag_omega,p,q);
    case 1   % GJR-Garch
        o = 1;
        [stdresid, error, mu, h] = core_gjrgarch(edata,parameters,flag_a0,ar,ma,flag_omega,p,q,o); 
    case 2   % EGarch
        o = 1;
        [stdresid, error, mu, h] = core_egarch(edata,parameters,flag_a0,ar,ma,flag_omega,p,q,o);
    otherwise
end

% eliminate the first data
t = (m + 1):T; 
Tau = T-m; 
        
switch errortype
    case 0   % Gaussion   
        LLF  =  0.5*(sum(log(h(t))) + sum(((edata(t)-mu(t)).^2)./h(t)) + Tau*log(2*pi)); 
        % LLF  =  sum(log(h(t))) + sum( ((edata(t)).^2)./h(t) );
        % LLF  =  0.5 * (LLF  +  Tau*log(2*pi));
    case 1   % Student's t
        n = parameters(length(parameters));
        LLF = Tau*gammaln(0.5*(n+1)) - Tau*gammaln(n/2) - Tau/2*log(pi*(n-2));
        LLF = LLF - 0.5*sum(log(h(t))) - ((n+1)/2)*sum(log(1 + stdresid(t).^2./(n-2)));
        LLF = -LLF;
    case 2   % Hansen's skewed t
        lam  = parameters(flag_a0+ar+ma+flag_omega+p+q+o+1);
        n = parameters(flag_a0+ar+ma+flag_omega+p+q+o+2); 
        LLF = skewtdis_LL([n;lam], stdresid(t));
        LLF = 0.5*sum(log(h(t))) + LLF;   
    case 3   % SGT
        lam = parameters(flag_a0+ar+ma+flag_omega+p+q+o+1);
        n   = parameters(flag_a0+ar+ma+flag_omega+p+q+o+2);
        k   = parameters(flag_a0+ar+ma+flag_omega+p+q+o+3);
        LLF = 0.5*sum(log(h(t))) + sgtdis_LL([lam;k;n],stdresid(t));  
    otherwise
end


