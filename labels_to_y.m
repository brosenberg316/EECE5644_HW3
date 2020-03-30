function y = labels_to_y(labels,C)
% Create y matrix for cross entropy
y = zeros(C,length(labels));
for ii = 1:length(labels)
    y(:,ii) = (1:C)' == labels(ii);
end