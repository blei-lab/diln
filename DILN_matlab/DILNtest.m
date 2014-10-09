function [VBparam,VBpoint] = DILNtest(Xcnt,Xid,Gam,p,beta,u,Kern,HDP,gamma,maxite)
% DILNTEST tests the model learned using DILN
%
% Written by: John Paisley, Princeton University, jpaisley@princeton.edu

maxite = 100;

M = length(Xid);
[K,D] = size(Gam);

perctest = .5;
XcntTest = {};
XidTest = {};
for m = 1:M
    numW = sum(Xcnt{m});
    numTest = floor(perctest*numW);
    [a,b] = sort(rand(1,numW));
    wordVec = [];
    for i = 1:length(Xid{m})
        wordVec = [wordVec Xid{m}(i)*ones(1,Xcnt{m}(i))];
    end
    wordTestVec = wordVec(b(1:numTest));
    wordTrainVec = wordVec(b(numTest+1:end));
    Xid{m} = unique(wordTrainVec);
    Xcnt{m} = histc(wordTrainVec,Xid{m});
    XidTest{m} = unique(wordTestVec);
    XcntTest{m} = histc(wordTestVec,XidTest{m});
end
        
N = zeros(K,M);                 % Counts matrix
Nm = zeros(M,1);
for m = 1:M
    Nm(m) = sum(Xcnt{m});
end
A = zeros(K,M) + 10;            % Posterior gamma distribution parameters
B = zeros(K,M) + 10;            %    ''
mu = 0*repmat(u,1,M);                % Posterior normal parameters
sig = 1*ones(K,M)*(1-HDP);        %    ''
% sig = repmat(diag(Kern),1,M)*(1-HDP);        %    ''

psiGam = psi(Gam);
psisumGam = psi(sum(Gam,2));
invKern = inv(Kern);

Lbound_const = M*K/2 + K*gammaln(gamma) - D*K*gammaln(gamma/D) ...
               - sum(gammaln(sum(Gam,2)) - sum(gammaln(Gam),2) + sum((Gam-gamma/D).*(psiGam - repmat(psisumGam,1,D)),2));
     	      % + K*log(alpha) - (alpha-1)*sum(log(1-V(1:end-1))) + (10^-6-1)*log(beta) - beta*10^-6;
[t1,t2,t3] = svd(Kern);
Lbound_const = Lbound_const - M/2*sum(log(diag(t2)));
bool = 1;
Lbound = 0;
ite = 0;
time = [];
llikall = [];
llikdocmean = [];
llikdocstd  = [];
while bool
    tic
    ite = ite + 1;
    Lbound(ite) = Lbound_const;
    llikall(ite) = 0;
    llikdocmean(ite) = 0;
    llikdocstd(ite) = 0;
    
  % Calculate latent membership probabilities, update Dirichlet parameters
  % and calculate lower bound
    lnZ = psi(A) - log(B);
    for m = 1:M
        C = psiGam(:,Xid{m}) + repmat(lnZ(:,m)-psisumGam,1,length(Xid{m}));
        C = C - repmat(max(C,[],1),K,1);
        C = exp(C);
        C = C./repmat(sum(C,1),K,1);
        N(:,m) = C*Xcnt{m}';
        Lbound(ite) = Lbound(ite) + sum(C.*psiGam(:,Xid{m}),1)*Xcnt{m}' - N(:,m)'*(psisumGam - lnZ(:,m));
        Lbound(ite) = Lbound(ite) - sum(C.*log(C+eps),1)*Xcnt{m}';
    end
    Lbound(ite) = Lbound(ite) + sum(sum(A - (A-1).*psi(A) - log(B) + gammaln(A)));
    Lbound(ite) = Lbound(ite) - sum(sum((A./B).*exp(mu + .5*sig)));
    if HDP ~= 1
        Lbound(ite) = Lbound(ite) + .5*sum(sum(log(sig)));
        for m = 1:M
            Lbound(ite) = Lbound(ite) - .5*(mu(:,m)-u)'*invKern*(mu(:,m)-u) - .5*diag(invKern)'*sig(:,m);
        end
    end
    Lbound(ite) = Lbound(ite) + beta*p'*sum(mu,2) - M*sum(gammaln(beta*p)) + (beta*p-1)'*sum(lnZ,2);
    
  % ============ Perform gradient ascent on other parameters ==============  
    expMUV = exp(mu + .5*sig);
    sumEZ = sum(A./B,1);
    A = N + beta*repmat(p,1,M);
    B = repmat(Nm'./sumEZ,K,1) + expMUV;
    Lbound(ite) = Lbound(ite) - (log(sumEZ) + (sum(A./B,1)-sumEZ)./sumEZ)*Nm;       

    
    if HDP ~= 1 & ite > 0
      % - Perform gradient ascent for posterior normal parameters
        for stepnum = 1:5 %(10*(ite==1) + 3*(ite > 1))
            for m = 1:M
                gradMU = beta*p - (A(:,m)./B(:,m)).*exp(mu(:,m) + .5*sig(:,m)) - invKern*(mu(:,m)-u);
                gradV = -.5*(A(:,m)./B(:,m)).*exp(mu(:,m) + .5*sig(:,m)) - .5*diag(invKern) + .5./sig(:,m);
                stepsize = getstepMUV(mu(:,m),sig(:,m),gradMU,gradV,beta*p,A(:,m)./B(:,m),u,invKern);
                mu(:,m) = mu(:,m) + stepsize*gradMU;
                sig(:,m) = sig(:,m) + stepsize*gradV;
                stepsizeall(m) = stepsize;
            end
        end                
    end
    time(ite) = toc;
    
%   % - Calculate performance measures
%     tic;
%     numSamp = 250;
%     tmp = zeros(numSamp,M);
%     if mod(ite,10) == 0
%         for samp = 1:numSamp
%             sampGam = gamrnd(Gam,1);
%             sampGam = sampGam./repmat(sum(sampGam,2),1,D);
%             weight = gamrnd(A,1./B);
%             weight = weight./repmat(sum(weight,1),K,1);
%             for m = 1:M
%                 tmp(samp,m) = XcntTest{m}*log(sampGam(:,XidTest{m})'*weight(:,m));
%             end
%         end
%     end
%     maxtmp = max(tmp,[],1);
%     tmp = tmp - repmat(maxtmp,numSamp,1);
%     tmp = maxtmp + log(sum(exp(tmp),1)) - log(numSamp);
%     llikall(ite) = sum(tmp)/sum(Nm);
%     perdocllik = tmp./Nm';
%     llikdocmean(ite) = mean(perdocllik);
%     llikdocstd(ite) = std(tmp./Nm');       
%     timeTest(ite) = toc;
    
    if ite > 25
        if abs(Lbound(ite) - Lbound(ite-1))/abs(Lbound(ite-1)) < 10^-3
            bool = 0;
        elseif ite == maxite
            bool = 0;
        end
    end
end

tic;
numSamp = 500;
tmp = zeros(numSamp,M);
for samp = 1:numSamp
    sampGam = gamrnd(Gam,1);
    sampGam = sampGam./repmat(sum(sampGam,2),1,D);
    weight = gamrnd(A,1./B);
    weight = weight./repmat(sum(weight,1),K,1);
    for m = 1:M
        tmp(samp,m) = XcntTest{m}*log(sampGam(:,XidTest{m})'*weight(:,m));
    end
end
maxtmp = max(tmp,[],1);
tmp = tmp - repmat(maxtmp,numSamp,1);
tmp = maxtmp + log(sum(exp(tmp),1)) - log(numSamp);
llikall(ite) = sum(tmp)/sum(Nm);
perdocllik = tmp./Nm';
llikdocmean(ite) = mean(perdocllik);
llikdocstd(ite) = std(tmp./Nm');
timeTest(ite) = toc;
VBparam = {perdocllik};
VBpoint = {Lbound,time,llikall,llikdocmean,llikdocstd,timeTest};