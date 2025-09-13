%% *GENERATION OF STRONG GROUND MOTION SIGNALS BY COUPLING PHYSICS-BASED ANALYSIS WITH ARTIFICIAL NEURAL NETWORKS*
% Original code by Filippo Gatti, modified by Victor Hernández
% DICA - Politecnico di Milano
% July 2025

%% *NOTES*
% _train_ann_justPSA_: function train ANN on PSA values
%% *N.B.*
% Need for:
% _trann_define_inout.m, trann_check_vTn.m,ANN MATLAB tool_
%% *REFERENCES*
% https://fr.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html
% http://www.cs.cmu.edu/afs/cs/Web/Groups/AI/html/faqs/ai/neural/faq.html
function train_ann_justPSA(varargin)
    %% *SET-UP*
    wd  = varargin{1};
    ann = varargin{2};
    dbn_name = varargin{3};
    net_ID  = varargin{4};
    n_LoopsANN = varargin{5};
    TransferLearning = varargin{6};
    add_distance = varargin{7};
    add_m = varargin{8};
    add_lndistance = varargin{9};
    separate_classes = varargin{10};
   
    
    if net_ID <10
       verNet = ['v0',num2str(net_ID)];
    elseif net_ID <100
       verNet = ['v',num2str(net_ID)]; 
    end

    % _load database_
    db = load(ann.dbn);
    db.nr  = size(db.SIMBAD,2);

    db.vTn = varargin{11};
    db.nT  = numel(db.vTn);

    % _define input/target natural periods_
    [inp.vTn,tar.vTn,inp.nT,tar.nT] = trann_define_inout(ann.TnC);
    % [inp2.vTn,tar2.vTn,inp2.nT,tar2.nT] = trann_define_inout(ann.TnC);
   
    % _check input/target natural periods with database_
    [inp.idx,tar.idx] = trann_check_vTn(inp,tar,db,1e-8);

    % _select class-compatible sites (EC8)_
    switch upper(ann.scl)
        case 'ALL'
            idx_cl = ones(db.nr,1);
            % idx2_cl = ones(db2.nr,1);
        case 'AB'
            ia1 = strcmpi('A',{db.SIMBAD(:).site_EC8});
            ia2 = strcmpi('A*',{db.SIMBAD(:).site_EC8});
            ia  = logical(ia1+ia2);
            ib1 = strcmpi('B',{db.SIMBAD(:).site_EC8});
            ib2 = strcmpi('B*',{db.SIMBAD(:).site_EC8});
            ib  = logical(ib1+ib2);
            idx_cl = logical(ia+ib);
        case 'CD'
            ia1 = strcmpi('C',{db.SIMBAD(:).site_EC8});
            ia2 = strcmpi('C*',{db.SIMBAD(:).site_EC8});
            ia  = logical(ia1+ia2);
            ib1 = strcmpi('D',{db.SIMBAD(:).site_EC8});
            ib2 = strcmpi('D*',{db.SIMBAD(:).site_EC8});
            ib  = logical(ib1+ib2);
            idx_cl = logical(ia+ib);
    end
    idx_cl1 = find(idx_cl==1);
    db.nr   = numel(idx_cl1);
    % idx2_cl1 = find(idx2_cl==1);
    % db2.nr   = numel(idx2_cl1);
    for i_=1:db.nr
        db.simbad(i_) = db.SIMBAD(idx_cl1(i_));
    end
    %  for i_=1:db2.nr
    %     db2.simbad(i_) = db2.SIMBAD(idx_cl1(i_));
    % end

    %% *DEFINE ANN INPUTS/TARGETS (PSA-T*)*
    PSA = -999*ones(db.nr,db.nT);
    % PSA2 = -999*ones(db2.nr,db2.nT);
    
    switch ann.cp
        % _Three COMPONENTS (separate branches)
        case {'h12v'}
            for j_ = 1:db.nr
                PSA_1(j_,:) = db.simbad(j_).psa_h1(:)';
                PSA_2(j_,:) = db.simbad(j_).psa_h2(:)';
                PSA_3(j_,:) = db.simbad(j_).psa_v(:)';
            end
        % _BOTH HORIZONTAL COMPONENTS (separate branches)
        case {'h12'}
            for j_ = 1:db.nr
                PSA_1(j_,:) = db.simbad(j_).psa_h1(:)';
                PSA_2(j_,:) = db.simbad(j_).psa_h2(:)';
            end
            % _HORIZONTAL COMPONENT 1_
        case {'h1'}
            for j_ = 1:db.nr
                PSA_1(j_,:) = db.simbad(j_).psa_h1(:)';
            end
            % for j_ = 1:db2.nr
            %     PSA2(j_,:) = db2.simbad(j_).psa_h1(:)';
            % end
            % _HORIZONTAL COMPONENT 2_
        case {'h2'}
            for j_ = 1:db.nr
                PSA_1(j_,:) = db.simbad(j_).psa_h2(:)';
            end
            % for j_ = 1:db2.nr
            %     PSA2(j_,:) = db2.simbad(j_).psa_h2(:)';
            % end
              % _HORIZONTAL ROTATIONAL INVARIANT
        case {'h_inv'}
            for j_ = 1:db.nr
                PSA_1(j_,:) = [db.simbad(j_).psa_inv(:)'];
                % PGV(j_,:) = db.simbad(j_).pgv(1);
            end
        case 'gh'
            for j_ = 1:db.nr
                PSA_1(j_,:) = geomean([db.simbad(1,j_).psa_h1(:)';...
                     db.simbad(1,j_).psa_h2(:)'],1);
            end
            % for j_ = 1:db2.nr
            %     PSA2(j_,:) = geomean([db2.simbad(1,j_).psa_h1(:)';...
            %         db2.simbad(1,j_).psa_h2(:)'],1);
            %end
            % _VERTICAL COMPONENT_
        case 'ud'
            for j_ = 1:db.nr
                PSA_1(j_,:) = db.simbad(j_).psa_v(:)';
            end
            % for j_ = 1:db2.nr
            %     PSA2(j_,:) = db2.simbad(j_).psa_v(:)';
            %end
    end
    
    %% *DEFINE INPUT/TARGET PSA POOL (LOG)*

     if strcmp(ann.cp,'h12')
        inp.simbad_1  = -999*ones(inp.nT,db.nr);
        tar.simbad_1  = -999*ones(tar.nT,db.nr);

        inp.simbad_2  = -999*ones(inp.nT,db.nr);
        tar.simbad_2 = -999*ones(tar.nT,db.nr);

        inp.simbad_5  = -999*ones(inp.nT,db.nr);
        tar.simbad_5 = -999*ones(tar.nT,db.nr);
    else
        inp.simbad_1  = -999*ones(inp.nT,db.nr);
        tar.simbad_1  = -999*ones(tar.nT,db.nr);
     end

    for i_=1:inp.nT
          inp.simbad_1(i_,1:db.nr) = log10(PSA_1(1:db.nr,inp.idx(i_))./100)';
          if strcmp(ann.cp,'h12v')
             inp.simbad_2(i_,1:db.nr) = log10(PSA_2(1:db.nr,inp.idx(i_))./100)';
             inp.simbad_5(i_,1:db.nr) = log10(PSA_3(1:db.nr,inp.idx(i_))./100)';
          elseif strcmp(ann.cp,'h12')
             inp.simbad_2(i_,1:db.nr) = log10(PSA_2(1:db.nr,inp.idx(i_))./100)';
          end
    end

    for i_=1:tar.nT
          tar.simbad_1(i_,1:db.nr) = log10(PSA_1(1:db.nr,tar.idx(i_))./100)';
          if strcmp(ann.cp,'h12v')
             tar.simbad_2(i_,1:db.nr) = log10(PSA_2(1:db.nr,tar.idx(i_))./100)';
             tar.simbad_3(i_,1:db.nr) = log10(PSA_3(1:db.nr,tar.idx(i_))./100)';
          elseif strcmp(ann.cp,'h12')
             tar.simbad_2(i_,1:db.nr) = log10(PSA_2(1:db.nr,tar.idx(i_))./100)';
          end
    end

    %% Add extra data
    index_extra=0;
% Add distance to input if add_distance='True'
    if strcmp(add_distance,'True')
         index_extra=index_extra+1;
        for j_ = 1:db.nr
                inp.simbad_3(index_extra,j_) =  (max(db.simbad(j_).Rjb,0.01));
        end
       
    end

    if strcmp(add_m,'True')
         index_extra=index_extra+1;
        for j_ = 1:db.nr
                 inp.simbad_3(index_extra,j_) = db.simbad(j_).Mw;
        end
    end

     if strcmp(add_lndistance,'True')
          index_extra=index_extra+1;
        for j_ = 1:db.nr
              inp.simbad_3(index_extra,j_) =  log10(max(db.simbad(j_).Rjb,0.01));
        end
     end

     n_classes=0;
     if strcmp(separate_classes,'True')
        for j_ = 1:db.nr
        AA=string(db.simbad(j_).site_EC8);
        if strcmp(AA,'A*')
        AA='A';
            elseif strcmp(AA,'B*')
        AA='B';
             elseif strcmp(AA,'C*')
        AA='C';
           elseif strcmp(AA,'D*')
        AA='D';
            elseif strcmp(AA,'E')
        AA='B';
        end
        Class(j_)=AA;
        end
        Class = categorical(Class);  % ensures categories are known
        inp.simbad_4 = onehotencode(Class,1); 
        n_classes = size(inp.simbad_4,1);
     end
     

    %% *DESIGN BASIC ANN*
    
    dsg.ntr=n_LoopsANN;   
    NNs = cell(dsg.ntr,1);
    prf.vld = -999*ones(dsg.ntr,1);
    out.prf = 0.0;
            
    for i_=1:dsg.ntr

                [dsg,layers] = train_ann_basics_h12(ann,db.nr,inp.nT,tar.nT,dsg,TransferLearning,add_distance,add_m,add_lndistance,separate_classes,n_classes,ann.cp);

                fprintf('ANN %u/%u: \n',i_,dsg.ntr);
                %% *DEFINE INPUTS/TARGETS*
                % _ALL INPUT/TARGET TRAINING VALUES_
if strcmp(ann.cp,'h12v')
            if index_extra>0 && n_classes>0
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)',inp.simbad_5(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)',inp.simbad_4(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)',tar.simbad_3(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)',inp.simbad_5(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)',inp.simbad_4(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)',tar.simbad_3(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)',inp.simbad_5(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)',inp.simbad_4(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)',tar.simbad_3(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsX1Trn_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsX1Trn_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsT1Trn_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsX1Trn_4,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsX1vld_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsX1vld_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsT1vld_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsX1vld_4,dsT1vld_1,dsT1vld_2,dsT1vld_3);  

          elseif (index_extra>0) && (n_classes==0)
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)',inp.simbad_5(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)',tar.simbad_3(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)',inp.simbad_5(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)',tar.simbad_3(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)',inp.simbad_5(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)',tar.simbad_3(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsX1Trn_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsT1Trn_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsX1Trn_3,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsX1vld_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsT1vld_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsX1vld_3,dsT1vld_1,dsT1vld_2,dsT1vld_3);  

          else 
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)',inp.simbad_5(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)',tar.simbad_3(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)',inp.simbad_5(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)',tar.simbad_3(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)',inp.simbad_5(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)',tar.simbad_3(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsX1Trn_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsT1Trn_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_5,dsT1Trn_1,dsT1Trn_2,dsT1Trn_3);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsX1vld_5 = arrayDatastore(inp.simbad_5(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsT1vld_3 = arrayDatastore(tar.simbad_3(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_5,dsT1vld_1,dsT1vld_2,dsT1vld_3);  

            end
elseif strcmp(ann.cp,'h12')
          if index_extra>0 && n_classes>0
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)',inp.simbad_4(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)',inp.simbad_4(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)',inp.simbad_4(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)'};
                
                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsX1Trn_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_3,dsX1Trn_4,dsT1Trn_1,dsT1Trn_2);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsX1vld_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_3,dsX1vld_4,dsT1vld_1,dsT1vld_2);  

          elseif (index_extra>0) && (n_classes==0)
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)'};
                
                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsX1Trn_3,dsT1Trn_1,dsT1Trn_2);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsX1vld_3,dsT1vld_1,dsT1vld_2);  

          else 
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_2(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)',tar.simbad_2(:,dsg.idx.trn)'};            
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_2(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)',tar.simbad_2(:,dsg.idx.vld)'};             
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_2(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)',tar.simbad_2(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_2,dsT1Trn_1,dsT1Trn_2);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_2 = arrayDatastore(inp.simbad_2(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsT1vld_2 = arrayDatastore(tar.simbad_2(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_2,dsT1vld_1,dsT1vld_2);  
          end

      else %not h12

            if index_extra>0 && n_classes>0
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)',inp.simbad_4(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)',inp.simbad_4(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)',inp.simbad_4(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsX1Trn_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsX1Trn_4,dsT1Trn_1);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsX1vld_4 = arrayDatastore(inp.simbad_4(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_3,dsX1vld_4,dsT1vld_1);  

          elseif (index_extra>0) && (n_classes==0)
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)',inp.simbad_3(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)'};          
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)',inp.simbad_3(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)'};            
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)',inp.simbad_3(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)'};
                
                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsX1Trn_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsX1Trn_3,dsT1Trn_1);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsX1vld_3 = arrayDatastore(inp.simbad_3(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsX1vld_3,dsT1vld_1);  

          else 
                NNs{i_}.inp.trn = {inp.simbad_1(:,dsg.idx.trn)'};
                NNs{i_}.tar.trn = {tar.simbad_1(:,dsg.idx.trn)'};            
                NNs{i_}.inp.vld = {inp.simbad_1(:,dsg.idx.vld)'};
                NNs{i_}.tar.vld = {tar.simbad_1(:,dsg.idx.vld)'};             
                NNs{i_}.inp.tst = {inp.simbad_1(:,dsg.idx.tst)'};
                NNs{i_}.tar.tst = {tar.simbad_1(:,dsg.idx.tst)'};

                dsX1Trn_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.trn)');
                dsT1Trn_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.trn)');
                dsTrn1 = combine(dsX1Trn_1,dsT1Trn_1);

                dsX1vld_1 = arrayDatastore(inp.simbad_1(:,dsg.idx.vld)');
                dsT1vld_1 = arrayDatastore(tar.simbad_1(:,dsg.idx.vld)');
                dsVld1= combine(dsX1vld_1,dsT1vld_1); 

            end
      end

                            options = trainingOptions('adam', ...     % Adam optimization
    'InitialLearnRate', 0.005, ...                 % Set the initial learning rate to 0.01 
    'MaxEpochs', 600, ...                         % Maximum number of epochs to train algorithm
    'ValidationData', dsVld1, ...           % Dataset to use as the validation set
    'MiniBatchSize', 100, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 3,...
    'ValidationPatience',9,...        %early stopping taking the validation loss as reference
    'Verbose', false, ...                          % Outputs information about training 
    'OutputNetwork','best-validation-loss',...
    'L2Regularization', 0.002...
);    
                
                %% *TRAINING ANN*
                % getting net and infos on training sets and performances
                fprintf('TRAINING...\n');
                % [NNs{i_}.net,NNs{i_}.trs] = ...
                %     train(NNs{i_}.inp.trn,NNs{i_}.tar.trn,layers,options);

                % analyzeNetwork(layers)
    if strcmp(ann.cp,'h12v')
        lossFcn = @(Y1,Y2,Y3,T1,T2,T3) mse(Y1,T1)/3 + mse(Y2,T2)/3 + mse(Y3,T3)/3;
    elseif strcmp(ann.cp,'h12')
        lossFcn = @(Y1,Y2,T1,T2) 0.5*mse(Y1,T1) + 0.5*mse(Y2,T2);
    else
        lossFcn = @(Y1,T1) 1*mse(Y1,T1) ;
    end

               [NNs{i_}.net,NNs{i_}.trs] = trainnet(dsTrn1,layers,lossFcn,options);

                
                %% *TEST/VALIDATE ANN PERFORMANCE*
                NNs{i_} = train_ann_valid(NNs{i_},TransferLearning,index_extra,n_classes,ann.cp);
                prf.trn(i_,1) = double(NNs{i_}.prf.trn);
                prf.vld(i_,1) = double(NNs{i_}.prf.vld);
                prf.tst(i_,1) = double(NNs{i_}.prf.tst);
                prf.r(i_,1) = double(NNs{i_}.prf.r);
                prf.mae(i_,1) = double(NNs{i_}.prf.mae);
    end
                 
    %% *COMPUTE BEST PERFORMANCE*
    NNs = trann_train_best_performance(NNs,prf,dsg,wd,dbn_name,verNet);
    
    %% *COMPUTE REGRESSIONS*
    trann_train_psa_performance(NNs,inp,tar,wd,dsg,dbn_name,verNet,ann.cp,TransferLearning);
    %% *OUTPUT*
     if strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True') && strcmp(separate_classes,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Rjb_Mw_logRjb_Site',dbn_name,verNet)),'NNs'); 
    elseif strcmp(add_distance,'True') && strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Rjb_Mw_logRjb',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_distance,'True') && strcmp(add_m,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Rjb_Mw',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_distance,'True') && strcmp(add_lndistance,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Rjb_logRjb',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_m,'True') && strcmp(add_lndistance,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Mw_logRjb',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_distance,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Rjb',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_lndistance,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'logRjb',dbn_name,verNet)),'NNs'); 
       elseif strcmp(add_m,'True')
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,'Mw',dbn_name,verNet)),'NNs');
       else
        save(fullfile(wd,sprintf('net_%u_%s_%s_%s_%s.mat',...
        round(ann.TnC*100),ann.scl,ann.cp,dbn_name,verNet)),'NNs');
       end
     return
end
