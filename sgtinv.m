function inv = sgtinv(u,lambda,nu,k)
step = 0.001;
for x = -20:step:0
    p = sgtcdf(x,lambda,nu,k);
    if(p>=u)
        inv = x-step;
        return
    end
end