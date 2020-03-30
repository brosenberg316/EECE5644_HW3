function model_order = gmm_model_order(data,delta)
% Determines GMM model order using 10-fold cross-validation
N = length(data); K = 10;
% Divide dataset into 10 folds
partition_indices = zeros(K,2);
partitions = ceil(linspace(0,N,K+1));
for k = 1:K
    partition_indices(k,:) = [partitions(k)+1, partitions(k+1)];
end
perf_vals = zeros(K,6);
% Using cross-validation on all folds of dataset
for k = 1:K
    % Get validation dataset
    idx_validate = [partition_indices(k,1):partition_indices(k,2)];
    % Using fold k as validation set
    d_validate = data(:,idx_validate);
    % l_validate = labels(idx_validate);
    % Get training dataset
    idx_train = ~ismember(1:N, idx_validate);
    d_train = data(:,idx_train);
    % Iterate through model orders
    for M = 1:6
        % Perform expectation maximization
        [alpha,mu,Sigma] = gmm_expectation_maximization(M,d_train,delta);
        % Measure performance against validation fold
        perf_vals(k,M) = measure_performance(alpha,mu,Sigma,d_validate);
    end
end
% Determine model order with maximum performance value
[~,model_order] = max(mean(perf_vals,1));