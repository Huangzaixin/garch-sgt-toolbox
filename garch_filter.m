function [parameters, stderrors, tstats, pvalues, LLF, stdresid, error, mu, h, AIC, BIC] = garch_filter(data,flag_a0,ar,ma,flag_omega,p,q,startingvals,options,model,distr)
% PURPOSE: estimate the parameters of GARCH-Constant SGT model
% INPUTS:
%        data      : return
%        flag_a0   : flag_a0=1, ARMA with a0; flag_a0=0, ARMA without a0 
%        ar, ma    : orders of conditional mean equation;    ar = 0,1,2; ma = 0,1,2;
%        p, q      : orders of conditional variance model;   p = 0,1,2;  q = 0,1,2;
%     startingvals : the initial values of the parameters
%        options   : options of fmincon function
%        model     : GARCH type
% RETURNS: 
%        parameters: the parameter set
%        stderrors : standard errors of parameters
%        tstats    : t statistics of parameters
%        pvalues   : p-values of parameters
%        LLF       : the maximum loglikelihood value of GARCH-SGT model
%        stdresid  : the standardized residual
%        error     : residual
%        h         : conditional variance
%        AIC       : value of Akaike Information Criterion
%        BIC       : value of Bayesian Information Criterion
% AUTHOR: Zaixin Huang
% EMAIL:  huangzaixin@gmail.com
%% -------------------------------------------------------------------------
T = length(data);
arma_order_sum = ar + ma;
m = max([ar,ma,p,q]);        % the max order of the ARMA and GARCH orders

%% garch type
if strcmp(model,'SGARCH')
    garchtype = 0;
    o = 0;
elseif strcmp(model,'GJR')
    garchtype = 1;
    o = 1;
elseif strcmp(model,'EGARCH')
    garchtype = 2;
    o = 1;
end

%% error type 
if strcmp(distr,'NORMAL') 
    errortype = 0;
elseif strcmp(distr,'T')
    errortype = 1;
elseif strcmp(distr,'HANSEN') 
    errortype = 2;
elseif strcmp(distr,'SGT') 
    errortype = 3;    
end

%% initiate starting values
if isempty(startingvals)
    startingvals = init_startingvals(data,flag_a0,ar,ma,flag_omega,p,q,o,garchtype,errortype); 
end

%% Standard GARCH
%  b0,b1,b2>0;  b1+b2<1;  1>lambda>-1;  k>0;  n>2.001
if garchtype == 0
    switch errortype
        case 0 
             A = [zeros(flag_a0+arma_order_sum,flag_a0+arma_order_sum) -eye(flag_omega+p+q); 
                  zeros(1,flag_a0+arma_order_sum+1) ones(1,p+q)];
             b = [zeros(1,flag_omega+p+q) 1];
             
             LB  = [ -ones(arma_order_sum + flag_a0,1);     zeros(1+p+q,1)];     
             UB  = [  ones(arma_order_sum + flag_a0,1); +Inf*ones(1+p+q,1)]; 
        case 1  
             A = [zeros(flag_omega+arma_order_sum,flag_a0+arma_order_sum) -eye(flag_omega+p+q) zeros(flag_omega+arma_order_sum,1); 
                  zeros(1,flag_a0+arma_order_sum+1) ones(1,p+q)  zeros(1,1)];
             b = [zeros(1,flag_omega+p+q) 1];
             
             LB  = [ -ones(arma_order_sum + flag_a0,1);     zeros(1+p+q,1);    2.001];     
             UB  = [  ones(arma_order_sum + flag_a0,1); +Inf*ones(1+p+q,1);     +Inf]; 
        case 2
             A = [zeros(flag_omega+arma_order_sum,flag_a0+arma_order_sum) -eye(flag_omega+p+q) zeros(flag_omega+arma_order_sum,2); 
                  zeros(1,flag_a0+arma_order_sum+1) ones(1,p+q)  zeros(1,2)];
             b = [zeros(1,flag_omega+p+q) 1];
             
             LB  = [ -ones(arma_order_sum + flag_a0,1);     zeros(1+p+q,1);     -1;   2.001];     
             UB  = [  ones(arma_order_sum + flag_a0,1); +Inf*ones(1+p+q,1);      1;    +Inf];         
        case 3  
             A = [zeros(flag_omega+p+q,flag_a0+arma_order_sum) -eye(flag_omega+p+q)  zeros(flag_omega+p+q,3);...
                  zeros(1,flag_a0+arma_order_sum)      0  ones(1,p)  ones(1,q)                   zeros(1,3)];          
             b = [zeros(1,flag_omega+p+q) 1];   

             LB  = [ -ones(arma_order_sum + flag_a0,1);     zeros(flag_omega+p+q,1);     -1;    2.001;     0];     
             UB  = [  ones(arma_order_sum + flag_a0,1); +Inf*ones(flag_omega+p+q,1);      1;     +Inf;  +Inf]; 
    end  
end

%% GJR-GARCH
%  (1) Omega > 0
%  (2) Alpha(i) >= 0 for i = 1,2,...P
%  (3) Beta(i)  >= 0 for i = 1,2,...Q
%  (4) sum(Alpha(i) + Beta(j) + 0.5*Tarch(z)) < 1 for i = 1,2,...P, j = 1,2,...Q and z = 1,2,...O(OU)
%  (5) Alpah(j) + Tarch(j) >=0 for j=1,2,...Q (In general, P=Q)
if garchtype ==1
      switch errortype
        case 0 
            A = [zeros(1+p+q,flag_a0 + arma_order_sum)  -eye(1+p+q,1+p+q)   zeros(1+p+q,o)     ;...
                 zeros(p,flag_a0 + arma_order_sum + 1)  -eye(p,p)   zeros(p,q)    -eye(p,o)    ;...       
                 zeros(1,flag_a0 + arma_order_sum)     0  ones(1,p)  ones(1,q)  0.5*ones(1,o) ];         
            b = [zeros(1,1+p+q+p) 1]; 
                 
            LB  = [-ones(arma_order_sum + flag_a0,1);      zeros(1+p+q,1);  -Inf*ones(o,1)];     
            UB  = [ ones(arma_order_sum + flag_a0,1);  +Inf*ones(1+p+q,1);  +Inf*ones(o,1)]; 
        case 1  
            A = [zeros(1+p+q,flag_a0 + arma_order_sum)  -eye(1+p+q,1+p+q)   zeros(1+p+q,o)     zeros(1+p+q,1);...
                 zeros(p,flag_a0 + arma_order_sum + 1)  -eye(p,p)   zeros(p,q)    -eye(p,o)        zeros(p,1);...       
                 zeros(1,flag_a0 + arma_order_sum)     0  ones(1,p)  ones(1,q)  0.5*ones(1,o)        0 ];        
            b = [zeros(1,1+p+q+p) 1]; 
                 
            LB  = [-ones(arma_order_sum + flag_a0,1);      zeros(1+p+q,1);  -Inf*ones(o,1);     2.001];     
            UB  = [ ones(arma_order_sum + flag_a0,1);  +Inf*ones(1+p+q,1);  +Inf*ones(o,1);     +Inf]; 
        case 2
            A = [zeros(1+p+q,flag_a0 + arma_order_sum)  -eye(1+p+q,1+p+q)   zeros(1+p+q,o)     zeros(1+p+q,2);...
                 zeros(p,flag_a0 + arma_order_sum + 1)  -eye(p,p)   zeros(p,q)    -eye(p,o)        zeros(p,2);...       
                 zeros(1,flag_a0 + arma_order_sum)     0  ones(1,p)  ones(1,q)  0.5*ones(1,o)        0  0  ];         
            b = [zeros(1,1+p+q+p) 1]; 
                 
            LB  = [-ones(arma_order_sum + flag_a0,1);      zeros(1+p+q,1);  -Inf*ones(o,1);     -1;    2.001];     
            UB  = [ ones(arma_order_sum + flag_a0,1);  +Inf*ones(1+p+q,1);  +Inf*ones(o,1);      1;     +Inf];         
        case 3    
            A = [zeros(1+p+q,flag_a0 + arma_order_sum)  -eye(1+p+q,1+p+q)   zeros(1+p+q,o)     zeros(1+p+q,3);...
                 zeros(p,flag_a0 + arma_order_sum + 1)  -eye(p,p)   zeros(p,q)    -eye(p,o)        zeros(p,3);...       
                 zeros(1,flag_a0 + arma_order_sum)     0  ones(1,p)  ones(1,q)  0.5*ones(1,o)        0  0  0];        
            b = [zeros(1,1+p+q+p) 1]; 
                 
            LB  = [-ones(arma_order_sum + flag_a0,1);      zeros(1+p+q,1);  -Inf*ones(o,1);     -1;    2.001;      0];     
            UB  = [ ones(arma_order_sum + flag_a0,1);  +Inf*ones(1+p+q,1);  +Inf*ones(o,1);      1;     +Inf;   +Inf]; 
      end
end

%% EGARCH
if garchtype == 2
      A = [];
      b = []; 
      switch errortype 
        case 0 
             LB  = [-ones(arma_order_sum + flag_a0,1);        -Inf*ones(1+p+q+o,1)];     
             UB  = [ ones(arma_order_sum + flag_a0,1);        +Inf*ones(1+p+q+o,1)];
        case 1  
             LB  = [-ones(arma_order_sum + flag_a0,1);        -Inf*ones(1+p+q+o,1);       0];     
             UB  = [ ones(arma_order_sum + flag_a0,1);        +Inf*ones(1+p+q+o,1);    +Inf];
        case 2   
             LB  = [-ones(arma_order_sum + flag_a0,1);        -Inf*ones(1+p+q+o,1);      -1;      0];     
             UB  = [ ones(arma_order_sum + flag_a0,1);        +Inf*ones(1+p+q+o,1);       1;   +Inf];           
        case 3   
             LB  = [-ones(arma_order_sum + flag_a0,1);        -Inf*ones(1+p+q+o,1);      -1;    2.001;     0];     
             UB  = [ ones(arma_order_sum + flag_a0,1);        +Inf*ones(1+p+q+o,1);       1;     +Inf;  +Inf];             
      end
end

%% options    
if isempty(options)
   options  =  optimset('fmincon');
   options  =  optimset(options , 'TolFun'      , 1e-7); 
   options  =  optimset(options , 'TolX'        , 1e-7); 
   options  =  optimset(options , 'TolCon'      , 1e-7); 
   % options  =  optimset(options , 'Algorithm' , 'sqp'); 
   options  =  optimset(options , 'Display'     , 'iter'); 
   options  =  optimset(options , 'Diagnostics' , 'on'); 
   options  =  optimset(options , 'LargeScale'  , 'off'); 
   options  =  optimset(options , 'MaxFunEvals' , 10000); 
end


%% expand data
stdEstimate = std(data,1);
edata  =  [stdEstimate(ones(m,1)); data];

%% Estimate the parameters
[parameters,LLF] = fmincon('garch_likelihood',startingvals,A,b,[],[],LB,UB,[],options,edata,flag_a0,ar,ma,flag_omega,p,q,o,m,garchtype,errortype);
LLF = -LLF;

%% get standerdized residuals and conditional variance
switch garchtype 
    case 0  % Basic GARCH  
        [stdresid, error, mu, h] = core_basicgarch(edata,parameters,flag_a0,ar,ma,flag_omega,p,q); 
    case 1  % GJR GARCH
        [stdresid, error, mu, h] = core_gjrgarch(edata,parameters,flag_a0,ar,ma,flag_omega,p,q,o); 
    case 2  % EGARCH 
        [stdresid, error, mu, h] = core_egarch(edata,parameters,flag_a0,ar,ma,flag_omega, p,q,o); 
end

t = (1+m):size(edata); 
stdresid = stdresid(t);
error = error(t);
mu = mu(t);
h = h(t);

%% calculate standard variance of the parameters
hess = hessian_2sided('garch_likelihood',parameters,edata,flag_a0,ar,ma,flag_omega,p,q,o,m,garchtype,errortype); 
stderrors = sqrt(diag(hess^(-1)));
tstats = parameters./stderrors;
pvalues = 2*(1-tcdf(abs(tstats), T-length(parameters)));   % p-values   

AIC = -2*LLF + 2*size(parameters,1);  
BIC = -2*LLF + size(parameters,1)*log(size(data,1));


