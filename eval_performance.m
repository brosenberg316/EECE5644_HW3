function score = eval_performance(H,labels)
[val,class_label] = max(H,[],1);
num_correct = sum(class_label == labels);
score = (num_correct/length(labels));
