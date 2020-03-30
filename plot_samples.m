function plot_samples(data,labels,C,title_str)
colors = rand(C,3);
figure(1), clf,
for l = 1:C
    ind_l = find(labels==l);
    plot(data(1,ind_l),data(2,ind_l),'.',...
        'MarkerFaceColor',colors(l,:),'DisplayName', ['Class' num2str(l)]); 
    axis equal, hold on,
end
legend('show');
title(title_str); xlabel('x_1'); ylabel('x_2');