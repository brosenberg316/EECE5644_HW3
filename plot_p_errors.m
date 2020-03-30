function plot_p_errors(p_errors,num_samples)
num_perceptrons = 1:length(p_errors);
figure();
h = bar(num_perceptrons,p_errors');
set(h,{'DisplayName'},{'Sigmoid','TanH'}')
legend('show');
title(sprintf('10-fold Cross-Validation for %d Samples',num_samples));
xlabel('Number of Perceptrons'); ylabel('Mean P(error)');
