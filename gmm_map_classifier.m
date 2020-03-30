function labels = gmm_map_classifier(data,C,alpha,mu,Sigma,priors)
% Estimates data labels using MAP classification for GMM
% Estimate posterior as prior x class-conditional pdf
posteriors = zeros(C,length(data));
for kk = 1:C
    % Evaluate posterior for each class using all samples
    posteriors(kk,:) = priors(kk)*evalGMM(data,alpha{kk},mu{kk},Sigma{kk});
end
% Use MAP classification - find the max posterior value for each sample
[~,labels] = max(posteriors,[],1);