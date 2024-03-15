function [startingvalues] = init_startingvals(data,flag_a0,ar,ma,omega_term,p,q,o,garchType,errortype)
%% mean equation
a0 = mean(data);  
a1 = 0.1*ones(ar,1)/ar;   a2 = 0.1*ones(ma,1)/ma; 

if flag_a0 == 1
    meaneq_terms = [a0; a1; a2]; 
else
    meaneq_terms = [    a1; a2];
end

%% variance equation
b1 = 0.1*ones(p,1)/p;  b2 = 0.1*ones(q,1)/q;    o_terms = 0.1*ones(o,1)/o;

switch garchType
    case 0 % Standard Garch
        b0 = (1-(sum(b1)+sum(b2)))*var(data);
        if isequal(omega_term,1)
           vareq_terms = [b0; b1; b2];
        else
           vareq_terms = [    b1; b2];
        end
    case 1 % GJR Garch
        b0 = (1-(sum(b1)+sum(b2)))*var(data);
        if isequal(omega_term,1)
           vareq_terms = [b0; b1; b2; o_terms];
        else
           vareq_terms = [    b1; b2; o_terms];
        end
    case 2 % EGarch
        b0 = (1-(sum(b1)+sum(b2)))*log(var(data)); 
        if isequal(omega_term,1)
           vareq_terms = [b0; b1; b2; o_terms];
        else
           vareq_terms = [    b1; b2; o_terms];
        end
end

%% error type
lam0 = -0.2;    n0 = 9;    k0 = 2;  

switch errortype
        case 0 % GAUSSIAN
            startingvalues = [meaneq_terms; vareq_terms];
        case 1 % STUDENT'S t
            startingvalues = [meaneq_terms; vareq_terms; n0];
        case 2 % Hansen's skew t
            startingvalues = [meaneq_terms; vareq_terms; lam0; n0];
        case 3 % SGT
            startingvalues = [meaneq_terms; vareq_terms; lam0; n0; k0];
end 

    
 
