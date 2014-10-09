function [VBparam,VBpoint] = DILN(Xcnt,Xid,D,K,HDP,gamma,maxite)
% DILN implements the discrete infinite logistic normal topic model
%
% Xcnt: A cell vector of length M.
%       Each cell represents a document and contains a counts of words.
% Xid:  A cell vector of length M.
%       Each cell contains the word id for the corresponding word in Xcnt.
% D: Vocabulary size
% K: Truncation level
% HDP: If HDP = 1, reduce to HDP, else run full algorithm
% maxite: maximum number of iterations
%
% Written by: John Paisley, Princeton University, jpaisley@princeton.edu

M = length(Xid);
alpha = 20;                     % Top-level concentration parameter
beta = 5;                       % Second-level concentration parameter
% gamma = round(1*D);             % Dirichlet base distribution parameter

g0 = ones(1,D)/D;

N = zeros(K,M);                 % Counts matrix
Nm = zeros(M,1);
for m = 1:M
    Nm(m) = sum(Xcnt{m});
end
A = zeros(K,M) + 10;            % Posterior gamma distribution parameters
B = zeros(K,M) + 10;            %    ''
mu = zeros(K,M);                % Posterior normal parameters
sig = 1*ones(K,M)*(1-HDP);   	%    ''
u = zeros(K,1);
Kern = 1*eye(K);                % Initialize kernel

for i = 1:length(Xid)
    tmp = zeros(D,1);
    tmp(Xid{i}) = Xcnt{i};
    tmp = tmp/sum(tmp);
    Xsparse(:,i) = sparse(tmp);
end
cent = KmeansL1(Xsparse,K,1);
Gam = 1 + gamma*full(cent') + rand(K,D)/10 + gamma/D;  % Posterior Dirichlet parameters
clear Xsparse cent;

V = ones(K,1);      % stick-breaking proportions
V(1) = 1/K;
for k = 2:K-1
    V(k) = (1/K)/prod(1-V(1:k-1));
end
p = V.*cumprod(1-[0;V(1:end-1)]);

Lbound_const = M*K/2 + K*gammaln(gamma) - D*K*gammaln(gamma/D);

bool = 1;
Lbound = 0;
ite = 0;
time = [];
while bool
    tic
    ite = ite + 1;
    Lbound(ite) = Lbound_const;
    
  % Calculate latent membership probabilities, update Dirichlet parameters and calculate lower bound
    psiGam = psi(Gam);
    psisumGam = psi(sum(Gam,2));
    Lbound(ite) = Lbound(ite) - sum(gammaln(sum(Gam,2)) - sum(gammaln(Gam),2) + sum((Gam-gamma/D).*(psiGam - repmat(psisumGam,1,D)),2));
    Gam = zeros(K,D) + repmat(gamma*g0,K,1); %gamma/D;
    lnZ = psi(A) - log(B);
    for m = 1:M
        C = psiGam(:,Xid{m}) + repmat(lnZ(:,m)-psisumGam,1,length(Xid{m}));
        C = C - repmat(max(C,[],1),K,1);
        C = exp(C);
        C = C./repmat(sum(C,1),K,1);
        N(:,m) = C*Xcnt{m}';
        Gam(:,Xid{m}) = Gam(:,Xid{m}) + C.*repmat(Xcnt{m},K,1);
        Lbound(ite) = Lbound(ite) + sum(C.*psiGam(:,Xid{m}),1)*Xcnt{m}' - N(:,m)'*(psisumGam - lnZ(:,m));
        Lbound(ite) = Lbound(ite) - sum(C.*log(C+eps),1)*Xcnt{m}';
    end
    Lbound(ite) = Lbound(ite) + sum(sum(A - (A-1).*psi(A) - log(B) + gammaln(A)));
    Lbound(ite) = Lbound(ite) - sum(sum((A./B).*exp(mu + .5*sig)));
    if HDP ~= 1
        Lbound(ite) = Lbound(ite) + .5*sum(sum(log(sig)));
        [t1,t2,t3] = svd(Kern);
        Lbound(ite) = Lbound(ite) - M/2*sum(log(diag(t2)));
        invKern = inv(Kern);
        for m = 1:M
            Lbound(ite) = Lbound(ite) - .5*(mu(:,m)-u)'*invKern*(mu(:,m)-u) - .5*diag(invKern)'*sig(:,m);
        end
    end
    Lbound(ite) = Lbound(ite) + beta*p'*sum(mu,2) - M*sum(gammaln(beta*p)) + (beta*p-1)'*sum(lnZ,2);
    Lbound(ite) = Lbound(ite) + K*log(alpha) - (alpha-1)*sum(log(1-V(1:end-1))) + (10^-6-1)*log(beta) - beta*10^-6;
    
  % Analytically update the posterior gamma paramters using auxiliary variable
    expMUV = exp(mu + .5*sig);
    sumEZ = sum(A./B,1);
    A = N + beta*repmat(p,1,M);
    B = repmat(Nm'./sumEZ,K,1) + expMUV;
    Lbound(ite) = Lbound(ite) - (log(sumEZ) + (sum(A./B,1)-sumEZ)./sumEZ)*Nm;
    
  % Update top-level stick proportions             
    for stepnum = 1:(20*(ite == 1) + 20*(ite > 1))
        one_V = 1 - V;
        sumMu = sum(mu,2);
        sumlnZ = sum(psi(A)-log(B),2);
        stickLeft = cumprod([1;one_V(1:end-1)]);
        p = V.*stickLeft;
        psiV = psi(beta*p);
        vVec = beta*stickLeft.*sumMu + beta*stickLeft.*sumlnZ;
      % Calculate gradient
        for k = 1:K-1
            tmp1 = beta*sum(sumMu(k+1:end).*p(k+1:end)/one_V(k));
            tmp2 = beta*sum(sumlnZ(k+1:end).*p(k+1:end)/one_V(k));
            tmp3 = M*beta*sum(psiV(k+1:end).*p(k+1:end)/one_V(k));
            vVec(k) = vVec(k) - tmp1 - tmp2;
            vVec(k) = vVec(k) + tmp3;
            vVec(k) = vVec(k) - M*beta*stickLeft(k)*psiV(k);
        end
        vVec(1:K-1) = vVec(1:K-1) - (alpha-1)./one_V(1:K-1);
        vVec(K) = 0;
        step_stick = getstepSTICK(V,vVec,sumMu,sumlnZ,beta,alpha,M);
        V = V + step_stick*vVec;
        p = V.*cumprod(1-[0;V(1:end-1)]);
    end
    
  % Optimize concentration parameters
    alpha = (10^-6+K-2)/(10^-6-sum(log(1-V(1:end-1))));
    for stepnum = 1:20
        dbeta = p'*sumMu - M*p'*psi(beta*p) + p'*sumlnZ + (10^-6 - 1)/beta - 10^-6;
        d2beta = -M*(p.^2)'*psi(1,beta*p) - (10^-6-1)/(beta^2);
        betaVec = -dbeta/d2beta;
        step_beta = getstepBeta(beta,betaVec,sumMu,sumlnZ,p,M);
        beta = beta + step_beta*betaVec;
    end

  % Perform gradient ascent for posterior normal parameters and update kernel
    if HDP ~= 1 && ite > 0
        for stepnum = 1:1
            for m = 1:M
                gradMU = beta*p - (A(:,m)./B(:,m)).*exp(mu(:,m) + .5*sig(:,m)) - invKern*(mu(:,m)-u);
                gradV = -.5*(A(:,m)./B(:,m)).*exp(mu(:,m) + .5*sig(:,m)) - .5*diag(invKern) + .5./sig(:,m);
                stepsize = getstepMUV(mu(:,m),sig(:,m),gradMU,gradV,beta*p,A(:,m)./B(:,m),u,invKern);
                mu(:,m) = mu(:,m) + stepsize*gradMU;
                sig(:,m) = sig(:,m) + stepsize*gradV;
            end
        end
        u = mean(mu,2);
        Kern = ((mu-repmat(u,1,M))*(mu-repmat(u,1,M))' + diag(sum(sig,2)))/M;
    end

    time(ite) = toc;
    if ite > 25
        if abs(Lbound(ite) - Lbound(ite-1))/abs(Lbound(ite-1)) < 10^-3
            bool = 0;
        elseif ite == maxite
            bool = 0;
        end
    end
end
VBparam = {Gam,p,u,Kern,alpha,beta,gamma};
VBpoint = {Lbound,time,HDP};