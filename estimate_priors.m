function priors = estimate_priors(labels,C)
% Estimate class priors from a dataset
priors = zeros(1,C);
for ii = 1:C
    priors(ii) = sum(labels == C)/numel(labels);
end