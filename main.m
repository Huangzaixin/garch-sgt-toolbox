%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Zaixin Huang
% Email:  huangzaixin@gmail.com
% Write for Chinese page "基于SGT分布的ES估计、后验分析及在沪深股市中应用"，数理统计与管理，2020，39(2)
% Download from https://github.com/Huangzaixin
% Date: 2023/09/17
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% read data
break;
clear
format longG
alldata = csvread('DATA/example.csv', 1, 1); 
data = alldata(:,1); 
hwidth = 70  % histogram width

%% EGARCH-SGT   
[parameters,stderrors,tstats,pvalues,LL,stdresid,error,mu,h,AIC,BIC] = garch_filter(data,  0,1,1,  1,1,1, [],[],'EGARCH','SGT');
AIC
parameters

%% Histogram
h1 = histogram(stdresid,hwidth,'Normalization','pdf'); hold on;  

%% SGT
lam = parameters(length(parameters)-2);
n = parameters(length(parameters)-1);
k = parameters(length(parameters));

x = -10:0.1:10;
y = sgtpdf(x,lam,n,k);
plot(x,y,'-r','linewidth',1.5); hold on;

xlabel('SGT distribution');
xlim([-6 6]);
ylim([0 0.6]);
set(gca,'ytick',[0:0.2:0.6]);
set(gca,'fontsize',12.5); 
set(gcf,'color','white'); 

T = length(stdresid);
stdresid_u = zeros(T,1);
for t=1:1:size(stdresid)
    stdresid_u(t,1) = sgtcdf(stdresid(t,1),lam,n,k);
end
 
x = sgtinv(0.05,lam,n,k);
VaR_sgt = mu + sqrt(h).*x


%% EGARCH-Hansen's skew t
[parameters,stderrors,tstats,pvalues,LL,stdresid,error,mu,h,AIC,BIC] = garch_filter(data,  0,1,1,  1,1,1, [],[],'EGARCH','HANSEN');
AIC 
parameters

h1 = histogram(stdresid,hwidth,'Normalization','pdf'); hold on; 
xlim([-6 6]);
ylim([0 0.6]);
set(gca,'ytick',[0:0.2:0.6]);
set(gca,'fontsize',12.5);

% Hansen's skew t
lambda = parameters(length(parameters)-1);
nu = parameters(length(parameters));

x = -10:0.1:10;
y = skewtpdf(x,nu,lambda);
plot(x,y,'-.m','linewidth',2); hold on;
xlabel('Skewed t distribution');
stdresid_u = skewtcdf(stdresid,lambda,nu);   
set(gcf,'Units','centimeters','Position',[8 8 45 6]);

x = skewtinv(0.05,lambda,nu);
VaR_skewt = mu + sqrt(h).*x

%%
dlmwrite('DATA/stdresid.csv',stdresid,'precision','%10.15f');  
dlmwrite('DATA/mu.csv',mu,'precision','%10.15f');  
dlmwrite('DATA/h.csv',h,'precision','%10.15f');  
dlmwrite('DATA/stdresid_u.csv',stdresid_u,'precision','%10.15f');  


