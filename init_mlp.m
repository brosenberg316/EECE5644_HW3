function net = init_mlp(num_perceptrons, fn, data, y)
% Initializes a MLP with given number of perceptrons
net = patternnet(num_perceptrons);
net.performFcn = 'crossentropy';
% net.input.processFcns = {'removeconstantrows','mapminmax'};
% Determine which activation function to use
if fn == 1
    % Sigmoid function
    net.layers{1}.transferFcn = 'logsig';
elseif fn == 2
    % TanH function
    net.layers{1}.transferFcn = 'tansig';
end
C = size(y,1);
nX = size(data,1);
% Configure MLP dimensions to inputs and outputs
net = configure(net,data,y);
% Initialize all biases to zero
net.b{1} = zeros(num_perceptrons,1);
net.b{2} = zeros(C,1);
% Input input layer weight initialization using Xavier
net.IW{1} = xavier_init(num_perceptrons,nX);
% Hidden layer weight initialization using Xavier
net.LW{2,1} = xavier_init(C,num_perceptrons);
% Set max epochs to 200,000
net.trainParam.epochs = 200000;
% Use scaled conjugate gradient
net.trainFcn = 'trainscg';
net.trainParam.showWindow = false;