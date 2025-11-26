%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*
% Original code by Filippo Gatti, modified by Victor Hernández
% DICA - Politecnico di Milano
% July 2025
%% *NOTES*
% _train_tv_sets_: function to select training and validation percentages
%% *N.B.*
% Need for:_randperm.m_
%% *REFERENCES*
% https://fr.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html
% Here a dataset is loaded and divided into two parts: 90% for designing networks and 10% for testing them all.


function [varargout] = trann_tv_sets(varargin)

rng("shuffle");
    %% *SET-UP*
    nr   = varargin{1};
    pv   = varargin{2};
    pt   = varargin{3};
    
    %% *DEFINE PERCENTAGES*
    Q1   = ceil(0.95*nr*pv);
    Q2   = ceil(0.95*nr*pt);
    Q3   = floor(0.95*nr)-Q1-Q2;
    
    idx.all(:,1) = randperm(nr);
    idx.vld      = idx.all(1:Q1,1);
    idx.tst      = idx.all(Q1+(1:Q2),1);
    idx.trn      = idx.all(Q1+Q2+1:end,1);
    
    %% *OUTPUT*
    varargout{1} = idx.trn;
    varargout{2} = idx.vld;
    varargout{3} = idx.tst;
    
    return
end
