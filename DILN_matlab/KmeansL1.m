function cent = KmeansL1(X,K,numIte)

[D,N] = size(X);
[a,b] = sort(rand(1,N));
cent = X(:,b(1:K));
id = zeros(1,N);

L1dist = zeros(K,N);
for ite = 1:numIte
    for n = 1:N
        [a,id(n)] = min(sum(abs(repmat(X(:,n),1,K) - cent),1));
    end
    for k = 1:K
        idx = find(id==k);
        if isempty(idx)
            cent(:,k) = X(:,b(1+floor(rand*N)));
        else
            cent(:,k) = mean(X(:,idx),2);
        end
    end
end

tmp = histc(id,1:K);
[a,b] = sort(tmp,'descend');
cent = cent(:,b);