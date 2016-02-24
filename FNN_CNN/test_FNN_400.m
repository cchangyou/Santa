
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
opts.flatten = true;
opts.gpu = gpu;

Ntrain  = 60000; %Subsample
Ntest   = 10000;

data      = MnistData(Ntrain,Ntest,opts);
actFunc   = 'ReLU'; 
inSize    = [28 28 1];
convSizes = []; % 0 layers
linSizes  = [400 400 data.outSize]
NNFactory = @() SimpleCNN(inSize,convSizes, linSizes, actFunc);

batchSize = 100; stepsize = 4e-11; %%% step size needs to take square to match the paper
maxEpochs = 100; % Experiment Parameters

%% Setup experiment
exps={};

ex = SimpleExperiment('Santa',NNFactory,data,@Santa);
ex.descentOpts.learningRate = stepsize;
ex.descentOpts.rrr = 0;
ex.descentOpts.learningRateDecay =  0.00;
ex.descentOpts.weightDecay= 1/Ntrain;
ex.descentOpts.RMSpropDecay = 0.1;
ex.descentOpts.epsilon=1e-5;
ex.descentOpts.N=Ntrain;
ex.descentOpts.batchSize = batchSize;
ex.saveInterval = -1;
ex.descentOpts.learningRateBlockDecay=0.5;
ex.descentOpts.learningRateBlock = maxEpochs*0.2*Ntrain/batchSize;
ex.descentOpts.burnin= 30*Ntrain/batchSize;
ex.descentOpts.decay_grad = 0.4;
ex.descentOpts.anne_rate = 0.7;
exps{end+1}=ex;


exps=runExperiments(exps,maxEpochs,false);


