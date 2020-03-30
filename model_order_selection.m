function [num_perceptrons,fn] =  model_order_selection(data, labels, C)
% Returns the optimal number of perceptrons in the first layer and
% activation function for trained MLP of a given dataset
y = labels_to_y(labels,C);
% Range of perceptrons to test
perceptron_range = 1:10;
nX = size(data,1);
N = size(data,2);
% 10-fold cross-validation
K = 10;

% Divide dataset into 10 folds
partition_indices = zeros(K,2);
partitions = ceil(linspace(0,N,K+1));
for k = 1:K
    partition_indices(k,:) = [partitions(k)+1, partitions(k+1)];
end

mean_p_errors = zeros(2,perceptron_range(end));
% Try all perceptron counts
for num_perceptrons = 1:perceptron_range(end)
    p_error = zeros(2,K);
    % Try both activation functions
    for activation_fn = 1:2
        % Using cross-validation on all folds of dataset
        for k = 1:K
            % Get validation dataset fold
            idx_validate = [partition_indices(k,1):partition_indices(k,2)];
            % Using fold k as validation set
            d_validate = data(:,idx_validate);
            l_validate = labels(idx_validate);
            
            % Get training dataset, which is rest of data
            idx_train = ~ismember(1:N, idx_validate);
            d_train = data(:,idx_train);
            l_train = labels(idx_train);
            y_train = y(:,idx_train);
            % Initialize the MLP
            net = init_mlp(num_perceptrons, activation_fn, d_train, y_train);
            % Train the MLP
            net = train(net,d_train,y_train);
            % Validate MLP performance against the validation fold
            outputs = net(d_validate);
            out_ind = vec2ind(outputs);
            % Keep track of P_error for each fold
            p_error(activation_fn,k) = sum(l_validate ~= out_ind)/numel(l_validate);
        end
        
    end
    mean_p_errors(:,num_perceptrons) = mean(p_error,2);
end
plot_p_errors(mean_p_errors,N);

% Find the best num perceptrons + activation function combo
[~,idx] = min(mean_p_errors,[],'all','linear');
[fn,num_perceptrons] = ind2sub([nX num_perceptrons(end)],idx);
% Print out model selection parameters for reporting
if fn == 1
    activation_fn_str = 'Sigmoid';
elseif fn == 2
    activation_fn_str = 'TanH';
end
fprintf('Model Order Selection for %d Samples\nNumber of Perceptrons: %d\nActivation Function: %s',...
    N,num_perceptrons,activation_fn_str);