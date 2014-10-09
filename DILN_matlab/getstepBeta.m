function step = getstepBeta(beta,betaVec,sumMu,sumlnZ,p,M);

% Get the maximum allowable step length
maxstep = -beta/betaVec;
if maxstep > 0
    maxstep = min(1,maxstep);
else
    maxstep = 1;
end

% Get stepsize checkpoints
if maxstep > 0
    step_check_vec = [ .125 .25 .375 .5 .625 .75 .875 ]*maxstep;
else
    step_check_vec = 0;
end

% Calculate objective for each stepsize
f = zeros(1,length(step_check_vec));
for ite = 1:length(step_check_vec)
    step_check = step_check_vec(ite);
    beta_check = beta + step_check*betaVec;
    f(ite) = beta_check*p'*sumMu - M*sum(gammaln(beta_check*p)) + beta_check*p'*sumlnZ + (10^-6-1)*log(beta_check) - 10^-6*beta_check;
end

% Pick best stepsize
if ~isempty(f)
    [a,b] = max(f);
    step = step_check_vec(b);
else
    step = 0;
end
% plot(step_check_vec,f);pause;