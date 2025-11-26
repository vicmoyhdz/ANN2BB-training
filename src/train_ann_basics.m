%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*
% _Editor: Filippo Gatti
% CentraleSupélec - Laboratoire MSSMat
% DICA - Politecnico di Milano
% Copyright 2016_
%% *NOTES*
% _train_ann_basics_: function to design basics ANN
%% *N.B.*
% Need for:_trann_tv_sets.m,ANN MATLAB tool_

function [varargout] = train_ann_basics(varargin)
    %% *SET-UP*
    ann = varargin{1};
    nbs = varargin{2};
    % dsg.train_strategy = varargin{3};
    NoInputs=varargin{3};
    NoOutputs=varargin{4};
    dsg = varargin{5};
    TransferLearning = varargin{6};
    add_distance = varargin{7};
    add_m = varargin{8};
    add_lndistance = varargin{9};
    if strcmp(TransferLearning,'True')
        nbs2 = varargin{10};
    end

    if strcmp(add_distance,'True')
        NoInputs=NoInputs+1;
    end

    if strcmp(add_m,'True')
        NoInputs=NoInputs+1;
    end

    if strcmp(add_lndistance,'True')
        NoInputs=NoInputs+1;
    end

    %% *CREATE BASE NETWORK (MLP)*
    % ANN name
    dsg.fnm = sprintf('net_%u_%s_%s_%s',round(ann.TnC*100),ann.scl,ann.cp,dsg.train_strategy);
    % _number of Hidden Neurons_
    dsg.nhn = ann.nhn;

    % Set up Division of Data for Training, Validation, Testing
    switch dsg.train_strategy
        
        case 'classic'
            
%             % _subdivide indexes_
%             dsg.net.divideFcn = 'dividerand';
            dsg.net.divideParam.trainRatio = 70/100;
            dsg.net.divideParam.valRatio   = 15/100;
            dsg.net.divideParam.testRatio  =  15/100;

            [dsg.idx.trn,dsg.idx.vld,dsg.idx.tst] = trann_tv_sets(nbs,dsg.net.divideParam.valRatio,...
                dsg.net.divideParam.testRatio); 
      if strcmp(TransferLearning,'True')
            [idx2.trn,idx2.vld,idx2.tst] = trann_tv_sets(nbs2,0.15,...
                0.15);
      end

            llayers = [
    sequenceInputLayer(NoInputs,"Normalization","zscore")
    % lstmLayer(ann.nhn)
    fullyConnectedLayer(25)
    tanhLayer
    % leakyReluLayer(1)
    fullyConnectedLayer(15)
    tanhLayer
    % fullyConnectedLayer(9)
    % % tanhLayer
    % leakyReluLayer(1)
    fullyConnectedLayer(NoOutputs)];
    % leakyReluLayer(1)];
            
        case 'bootstrap'
            
            % _subdivide indexes_
            dsg.net.divideFcn = 'divideind';
            divideParam.trainRatio = 70/100;
            divideParam.valRatio   = 15/100;
            divideParam.testRatio  = 15/100;
            % base set of training-validation-test
            [dsg.trn_idx,dsg.net.divideParam.valInd,...
                dsg.net.divideParam.testInd] = dividerand(nbs,...
                divideParam.trainRatio,divideParam.valRatio,...
                divideParam.testRatio);
            
    end
    %% *OUTPUT*
    varargout{1} = dsg;
    varargout{2} = layers;
 if strcmp(TransferLearning,'True')
    varargout{3}=idx2;
 end
    return
end
