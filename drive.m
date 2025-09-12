%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*
% Original code by Filippo Gatti, modified by Victor Hernández
% DICA - Politecnico di Milano
% July 2025

clc
clear
close all
 
addpath src\ %subfolder with routines
folder_save='ANNs'; %folder to save the ouput ANNs

%% *TRAIN SET-UP (CUSTOMIZE)*
% *DATABASE SELECTED*

TransferLearning = 'False'; %if True you need to define also dbn_name2, otherwise dbn_name2 is ignored
dbn_name = 'ESM_SIMBADs';
dbn_name2 = 'ESM_SIMBADs'; %Database for transfer learning (TL). Put the same if not using TL
%The data sets should be located inside subfolder database

% *DEFINE THE NUMBER OF NETS TO BE TRAINED*
num_nets = 10; %number of individual nets
n_LoopsANN = 1; %number of trained nets before choosing the best one
add_distance = 'True';
add_m = 'True';
add_lndistance = 'True';
separate_classes = 'False';

%% *DEFINE TRAIN METADATA (CUSTOMIZE)*

% *ANN METADATA ann*
% _number of ann
ann.trn.nr = 2; %2 for horizontal (h12) and vertical (ud) components 
% _corner periods_
TnC = [1,1];
% _direction (ud=vertical;h12=both horizontal)
cp  = {'h12','ud'};
% _site class (ALL,AB,CD)_
scl = {'ALL','ALL'}; %ALL to use all site classes
% _number of neurons_
nnr = [20,20]; %not used

%%

% *WORKDIR*
% _main workdir_
wd = strcat(cd,'\',folder_save);
% _save path_
dbn = strcat(cd,'\database\',dbn_name,'.mat');
dbn2 = strcat(cd,'\database\',dbn_name2,'.mat');

if exist(wd,'dir')~=7
    wd = strcat(cd,'\',folder_save);   
    dbn = strcat(cd,'\database\',dbn_name,'.mat');
    dbn2 = strcat(cd,'\database\',dbn_name2,'.mat');
end

ann.trn.wd = fullfile(wd);
fprintf('Training Workdir: %s\n',ann.trn.wd);
fprintf('Training Database: %s\n',dbn);

for i_=1:ann.trn.nr
    ann.trn.mtd(i_).TnC = TnC(i_);
    ann.trn.mtd(i_).cp  = cp{i_};
    ann.trn.mtd(i_).scl = scl{i_};
    ann.trn.mtd(i_).nhn = nnr(i_);
end
clear TnC cp scl

% _database_
for i_ = 1:ann.trn.nr
    ann.trn.mtd(i_).dbn = dbn;
end

%% Training
tic
for iNet = 1:num_nets    
     net_ID = iNet;
for i_ = 1:ann.trn.nr
    if strcmp(TransferLearning,'True')
        train_ann_justPSA_TransferLearning(ann.trn.wd,ann.trn.mtd(i_),dbn_name,net_ID,n_LoopsANN,TransferLearning,dbn2,add_distance,add_m,add_lndistance,separate_classes);
    else
        train_ann_justPSA(ann.trn.wd,ann.trn.mtd(i_),dbn_name,net_ID,n_LoopsANN,TransferLearning,add_distance,add_m,add_lndistance,separate_classes);
    end
end
end
toc