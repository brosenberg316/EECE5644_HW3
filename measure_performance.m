function perf = measure_performance(alpha,mu,Sigma,data)
perf = (1/length(data))*sum(log(evalGMM(data,alpha,mu,Sigma)));
end
