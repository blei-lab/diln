function step = getstepSTICK(curr,grad,sumMu,sumlnZ,beta,alpha,M)

% Get the maximum allowable step length
step_zero = -curr(1:length(curr)-1)./grad(1:length(curr)-1);
step_one = (1-curr(1:length(curr)-1))./grad(1:length(curr)-1);
min_zero = min([1;step_zero(step_zero > 0)]);
min_one = min([1;step_one(step_one > 0)]);
max_step = min(min_zero,min_one);

% Get stepsize checkpoints
if max_step > 0
    step_check_vec = [ .125 .25 .375 .5 .625 .75 .875 ]*max_step;
else
    step_check_vec = 0;
end


% Calculate objective for each stepsize
f = zeros(1,length(step_check_vec));
for ite = 1:length(step_check_vec)
    step_check = step_check_vec(ite);
    vec_check = curr + step_check*grad;
    p = vec_check.*cumprod(1-[0;vec_check(1:end-1)]);
    f(ite) = beta*sumMu'*p - M*sum(gammaln(beta*p)) + (beta*p-1)'*sumlnZ + (alpha-1)*sum(log(1-vec_check(1:end-1)));
end

% Pick best stepsize
if ~isempty(f)
    [a,b] = max(f);
    step = step_check_vec(b);
else
    step = 0;
end

if b == 1
    rho = .5;
    bool = 1;
    fold = f(b);
    while bool
        step = rho*step;
        vec_check = curr + step*grad;
        p = vec_check.*cumprod(1-[0;vec_check(1:end-1)]);
        fnew = beta*sumMu'*p - M*sum(gammaln(beta*p)) + (beta*p-1)'*sumlnZ + (alpha-1)*sum(log(1-vec_check(1:end-1)));
         if fnew > fold
             fold = fnew;
         else
             bool = 0;
         end
%          f = [fnew f];
%          step_check_vec = [step step_check_vec];
    end
    step = step/rho;
end
% plot(step_check_vec,f);title(num2str(step));pause;