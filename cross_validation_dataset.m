function data = cross_validation_dataset(data,K)
N = size(data,2);
K = round(sqrt(N));

dummy = ceil(linspace(0,N,K+1));
for k = 1:K
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end

% Using fold k as validation set
indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
x_validate = x(indValidate);
y_validate = y(indValidate);

% Put rest of data into training set
if k == 1
    % If 1 fold, use remaining data 
    indTrain = [indPartitionLimits(k,2)+1:N];
elseif k == K
    indTrain = [1:indPartitionLimits(k,1)-1];
else
    indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
end
x_train = x(indTrain); % using all other folds as training set
y_train = y(indTrain);
% Ntrain = length(indTrain); Nvalidate = length(indValidate);

% b_idx = randi([1 length(x)],1,num_samples);
% data = data(:,b_idx);
% v = randGaussian(length(data),[0;0],1e-8*eye(2,2));
% data = data + v;

v = rand(8812,1);
b = 20; % block size
n = numel(v);
c = mat2cell(v,diff([0:b:n-1,n]));
z = cellfun(@median,c);
