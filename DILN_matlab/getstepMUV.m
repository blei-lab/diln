function stepsize = getstepMUV(currMu,currV,vecMu,vecV,bp,AdivB,u,invKern)

% Get the maximum allowable step length
steps = -currV./vecV;
isbound = (sum(steps > 0) > 0);
maxstep2 = min(steps(steps > 0));
if sum(steps >= 0) > 0
    maxstep = min([steps(steps > 0);1]);
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
    mu_check = currMu + step_check*vecMu;
    v_check = currV + step_check*vecV;
    f(ite) = mu_check'*bp - AdivB'*exp(mu_check + .5*v_check) - .5*(mu_check-u)'*invKern*(mu_check-u) - .5*diag(invKern)'*v_check + .5*sum(log(v_check));
end

% Pick best stepsize
if ~isempty(f)
    [a,b] = max(f);
    stepsize = step_check_vec(b);
else
    stepsize = 0;
end

if b == length(step_check_vec)
   rho = 1.5;
   bool = 1;
   fold = f(b);
   while bool
        stepsize = rho*stepsize;
        if isbound
           if stepsize > maxstep2
               bool = 0;
               break;
           end
        end
        mu_check = currMu + stepsize*vecMu;
        v_check = currV + stepsize*vecV;
        fnew = mu_check'*bp - AdivB'*exp(mu_check + .5*v_check) - .5*(mu_check-u)'*invKern*(mu_check-u) - .5*diag(invKern)'*v_check + .5*sum(log(v_check));
        if fnew > fold
            fold = fnew;
        else
            bool = 0;
        end
    end
    stepsize = stepsize/rho;
end

if b == 1
    rho = .5;
    bool = 1;
    fold = f(b);
    while bool
        stepsize = rho*stepsize;
        mu_check = currMu + stepsize*vecMu;
        v_check = currV + stepsize*vecV;
        fnew = mu_check'*bp - AdivB'*exp(mu_check + .5*v_check) - .5*(mu_check-u)'*invKern*(mu_check-u) - .5*diag(invKern)'*v_check + .5*sum(log(v_check));
         if fnew > fold
             fold = fnew;
         else
             bool = 0;
         end
    end
    stepsize = stepsize/rho;
end