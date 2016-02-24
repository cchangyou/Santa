
clearvars; clc; close all
addpath(genpath('.'));

global gpu;
% try
%     gpuDevice;
%     gpu=true;
% catch
%     gpu = false;
% end
gpu=false;

opts.precision = @single;
opts.flatten = false;
opts.gpu = gpu;

Ntrain  = 60000; %Subsample
Ntest   = 10000;

data      = MnistData(Ntrain,Ntest,opts);
actFunc   = 'ReLU';
inSize    = [28 28 1]; 

convSizes = ...
    [5   5 ;       % filter width
    5    5  ;       % filter height
    32   64;       % #out channels
    2   2   ;       % max pooling width
    2   2  ]';     % max pooling height
%1st    2nd           layers

linSizes  = [500 data.outSize];
NNFactory = @() SimpleCNN(inSize,convSizes, linSizes, actFunc);

batchSize = 100; stepsize = 4e-11;
maxEpochs = 100;

%% Setup experiment
exps={};
ex = SimpleExperiment('Santa',NNFactory,data,@Santa);
ex.descentOpts.learningRate = stepsize;
ex.descentOpts.learningRateDecay = 0.00;
ex.descentOpts.weightDecay = 1/Ntrain;
ex.descentOpts.RMSpropDecay = 0.8;
ex.descentOpts.epsilon = 1e-5;
ex.descentOpts.N = Ntrain;
ex.descentOpts.batchSize = batchSize;
ex.saveInterval = -1;
ex.descentOpts.learningRateBlockDecay = 0.5;
ex.descentOpts.learningRateBlock = maxEpochs*0.2*Ntrain/batchSize;
ex.descentOpts.burnin = 0.5*Ntrain/batchSize;
ex.descentOpts.decay_grad = 0.4;
ex.descentOpts.anne_rate = 0.8;
exps{end+1} = ex;

exps=runExperiments(exps,maxEpochs,false);
