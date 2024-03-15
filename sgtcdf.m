function cdf = sgtcdf(a,lambda, nu, k)
cdf = integral(@(x) sgtpdf(x,lambda,nu,k),-Inf,a);