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

if strcmp(component,'h12')
    branches=2;
else
    branches=1;
end
%branch 1
input1=featureInputLayer(NoInputs,"Normalization","zscore",Name="input1");
Branch1 = [input1,fullyConnectedLayer(20, 'Name', 'fc_1'),...
    tanhLayer('Name', 'tanh_1')];

%branch 2
if strcmp(component,'h12')
    input2=featureInputLayer(NoInputs,"Normalization","zscore",Name="input2");
    Branch2 = [input2,fullyConnectedLayer(20, 'Name', 'fc_2'),...
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


if strcmp(component,'h12')
    concat=concatenationLayer(1,branches,Name="concat");
    % shared = [fullyConnectedLayer(30, 'Name', 'fc_shared1'),...
    % tanhLayer('Name', 'tanh_shared1'),fullyConnectedLayer(30, 'Name', 'fc_shared2'),...
    %         tanhLayer('Name', 'tanh_shared2')];
    shared = [fullyConnectedLayer(40, 'Name', 'fc_shared1'),...
        tanhLayer('Name', 'tanh_shared1')];
elseif branches>1
    concat=concatenationLayer(1,branches,Name="concat");
    % shared = [fullyConnectedLayer(20, 'Name', 'fc_shared1'),...
    % tanhLayer('Name', 'tanh_shared1'),fullyConnectedLayer(25, 'Name', 'fc_shared2'),...
    %         tanhLayer('Name', 'tanh_shared2')];
    shared = [fullyConnectedLayer(20, 'Name', 'fc_shared1'),...
        tanhLayer('Name', 'tanh_shared1')];
else
    shared = [fullyConnectedLayer(20, 'Name', 'fc_shared1'),...
        tanhLayer('Name', 'tanh_shared1')];
end

% Outputs
output1 = fullyConnectedLayer(NoOutputs, 'Name', 'output1');
if strcmp(component,'h12')
    output2 = fullyConnectedLayer(NoOutputs, 'Name', 'output2');
end

layers = dlnetwork;
layers = addLayers(layers, Branch1);
if strcmp(component,'h12')
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
end

% Connect branches
if strcmp(component,'h12')
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
