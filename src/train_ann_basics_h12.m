%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*
% New ANN architecture by Victor Hernández
% DICA - Politecnico di Milano
% July 2025
%% *NOTES*
% _train_ann_basics_: function to design basics ANN
%% *N.B.*
% Need for:_trann_tv_sets.m,ANN MATLAB tool_

function [varargout] = train_ann_basics_h12(varargin)
%% *SET-UP*
ann = varargin{1};
nbs = varargin{2};
NoInputs=varargin{3};
NoOutputs=varargin{4};
dsg = varargin{5};
TransferLearning = varargin{6};
add_distance = varargin{7};
add_m = varargin{8};
add_lndistance = varargin{9};
separate_classes=varargin{10};
n_classes=varargin{11};
component=varargin{12};

if strcmp(TransferLearning,'True')
    nbs2 = varargin{13};
end

iextra=0;
if strcmp(add_distance,'True')
    iextra=iextra+1;
end

if strcmp(add_m,'True')
    iextra=iextra+1;
end

if strcmp(add_lndistance,'True')
    iextra=iextra+1;
end

n_first_level=ann.nhn(1);
if strcmp(component,'h12v')
    n_common=3*ann.nhn(2);
elseif strcmp(component,'h12')
    n_common=2*ann.nhn(2);
end

%% *CREATE BASE NETWORK (MLP)*
% ANN name
dsg.fnm = sprintf('net_%u_%s_%s_%s',round(ann.TnC*100),ann.scl,ann.cp);
% _number of Hidden Neurons_
dsg.nhn = ann.nhn;

% Set up Division of Data for Training, Validation, Testing

%             % _subdivide indexes_
%             dsg.net.divideFcn = 'dividerand';
dsg.net.divideParam.trainRatio = 70/100;
dsg.net.divideParam.valRatio   = 15/100;
dsg.net.divideParam.testRatio  =  15/100;
if strcmp(component,'h')
    [dsg.idx.trn,dsg.idx.vld,dsg.idx.tst] = trann_tv_sets(2*nbs,dsg.net.divideParam.valRatio,...
        dsg.net.divideParam.testRatio);
else
    [dsg.idx.trn,dsg.idx.vld,dsg.idx.tst] = trann_tv_sets(nbs,dsg.net.divideParam.valRatio,...
        dsg.net.divideParam.testRatio);
end
if strcmp(TransferLearning,'True')
    if strcmp(component,'h')
        [idx2.trn,idx2.vld,idx2.tst] = trann_tv_sets(2*nbs2,0.15,0.15);
    else
        [idx2.trn,idx2.vld,idx2.tst] = trann_tv_sets(nbs2,0.15,0.15);
    end
end

if strcmp(component,'h12v')
    branches=3;
elseif strcmp(component,'h12')
    branches=2;
else
    branches=1;
end
%branch 1
input1=featureInputLayer(NoInputs,"Normalization","zscore",Name="input1");
Branch1 = [input1,fullyConnectedLayer(n_first_level, 'Name', 'fc_1'),...
    tanhLayer('Name', 'tanh_1')];

%branches 2 & 5
if strcmp(component,'h12v')
    input2=featureInputLayer(NoInputs,"Normalization","zscore",Name="input2");
    Branch2 = [input2,fullyConnectedLayer(n_first_level, 'Name', 'fc_2'),...
    tanhLayer('Name', 'tanh_2')];

    input5=featureInputLayer(NoInputs,"Normalization","zscore",Name="input5");
    Branch5 = [input5,fullyConnectedLayer(n_first_level, 'Name', 'fc_5'),...
    tanhLayer('Name', 'tanh_5')];
%branch 2
elseif strcmp(component,'h12')
    input2=featureInputLayer(NoInputs,"Normalization","zscore",Name="input2");
    Branch2 = [input2,fullyConnectedLayer(n_first_level, 'Name', 'fc_2'),...
    tanhLayer('Name', 'tanh_2')];
end

if iextra>0
    %branch 3
    input3=featureInputLayer(iextra,"Normalization","zscore",Name="input3");
    extraBranch = [input3,fullyConnectedLayer(iextra*2, 'Name', 'fc_3'),...
    tanhLayer('Name', 'tanh_3')];
    branches=branches+1;
end

if n_classes>0
    catInput = featureInputLayer(4, 'Name', 'categoryInput');
    catBranch = [ catInput,fullyConnectedLayer(5, 'Name', 'input4'),...
     reluLayer('Name', 'cat_relu1')];
    branches=branches+1;
end

%shared
if strcmp(component,'h12') | strcmp(component,'h12v')
    concat=concatenationLayer(1,branches,Name="concat");
    shared = [fullyConnectedLayer(n_common, 'Name', 'fc_shared1'),...
    tanhLayer('Name', 'tanh_shared1')];
elseif branches>1
    concat=concatenationLayer(1,branches,Name="concat");
    shared = [fullyConnectedLayer(n_first_level, 'Name', 'fc_shared1'),...
    tanhLayer('Name', 'tanh_shared1')];
else
    shared = [fullyConnectedLayer(n_first_level, 'Name', 'fc_shared1'),...
    tanhLayer('Name', 'tanh_shared1')];
end

% Outputs
output1 = fullyConnectedLayer(NoOutputs, 'Name', 'output1');
if strcmp(component,'h12v')
    output2 = fullyConnectedLayer(NoOutputs, 'Name', 'output2');
    output3 = fullyConnectedLayer(NoOutputs, 'Name', 'output3');
elseif strcmp(component,'h12')
    output2 = fullyConnectedLayer(NoOutputs, 'Name', 'output2');
end

layers = dlnetwork;
layers = addLayers(layers, Branch1);
if strcmp(component,'h12v')
    layers = addLayers(layers, Branch2);
    layers = addLayers(layers, Branch5);
elseif strcmp(component,'h12')
    layers = addLayers(layers, Branch2);
end
if iextra>0
    layers = addLayers(layers, extraBranch);
end
if n_classes>0
    layers = addLayers(layers, catBranch);
end
if branches>1
    layers = addLayers(layers, concat);
end
layers = addLayers(layers, shared);
layers = addLayers(layers, output1);
if strcmp(component,'h12')
    layers = addLayers(layers, output2);
elseif strcmp(component,'h12v')
    layers = addLayers(layers, output2);
    layers = addLayers(layers, output3);
end

% Connect branches
if strcmp(component,'h12v')
    layers = connectLayers(layers, 'tanh_1', 'concat/in1');
    layers = connectLayers(layers, 'tanh_2', 'concat/in2');
    layers = connectLayers(layers, 'tanh_5', 'concat/in3');
    if iextra>0
        layers = connectLayers(layers, 'tanh_3', 'concat/in4');
    end
    if n_classes>0
        layers = connectLayers(layers, 'cat_relu1', 'concat/in5');
    end
    layers = connectLayers(layers, 'concat', 'fc_shared1');
    layers = connectLayers(layers, 'tanh_shared1', 'output1');
    layers = connectLayers(layers, 'tanh_shared1', 'output2');
    layers = connectLayers(layers, 'tanh_shared1', 'output3');
elseif strcmp(component,'h12')
    layers = connectLayers(layers, 'tanh_1', 'concat/in1');
    layers = connectLayers(layers, 'tanh_2', 'concat/in2');
    if iextra>0
        layers = connectLayers(layers, 'tanh_3', 'concat/in3');
    end
    if n_classes>0
        layers = connectLayers(layers, 'cat_relu1', 'concat/in4');
    end
    layers = connectLayers(layers, 'concat', 'fc_shared1');
    layers = connectLayers(layers, 'tanh_shared1', 'output1');
    layers = connectLayers(layers, 'tanh_shared1', 'output2');
else %not h12
    if branches>1
        layers = connectLayers(layers, 'tanh_1', 'concat/in1');
        if iextra>0
            layers = connectLayers(layers, 'tanh_3', 'concat/in2');
        end
        if n_classes>0
            layers = connectLayers(layers, 'cat_relu1', 'concat/in3');
        end
        layers = connectLayers(layers, 'concat', 'fc_shared1');
        layers = connectLayers(layers, 'tanh_shared1', 'output1');
    else
        layers = connectLayers(layers, 'tanh_1', 'fc_shared1');
        layers = connectLayers(layers, 'tanh_shared1', 'output1');
    end
end

% figure; plot(layers)


%% *OUTPUT*
varargout{1} = dsg;
varargout{2} = layers;
if strcmp(TransferLearning,'True')
    varargout{3}=idx2;
end
return
end
