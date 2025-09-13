function trann_train_psa_performance(varargin)
    %% *SET-UP*
    ann = varargin{1};
    inp = varargin{2};
    tar = varargin{3};
    wd  = varargin{4};
    dsg = varargin{5};
    dbn_name = varargin{6};
    verNet = varargin{7};
    component= varargin{8}; 
    TransferLearning = varargin{9};    
    
    plot_set_up;    
    TnC  = inp.vTn(1);    
    xpl = cell(3,1);
    ypl = cell(3,1);    
    nT=length(tar.vTn(:));
    %% *DEFINE LIMITS*
    % _TRAINING SET_ 
    % xlm =  [0.00;1.00];
    ylm =  [-0.75;0.75];
    xtk =  0.00:0.25:1.00;
    ytk = -1.00:0.25:1.00;
    xlm = [-0.05;1];
    
    %% *COMPUTE ERROR BARS*
    % _TRAINING SET_ 
    %
    xpl{2,1} = tar.vTn(:)./TnC;
    xpl{3,1} = tar.vTn(:)./TnC;
    xpl{4,1} = tar.vTn(:)./TnC;
    %
    %Perfomance plots
    if strcmp(component,'h12v')
        if strcmp(TransferLearning,'True')
            ypl{2,1} = mean([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],1);
            ypl{3,1} = mean([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2},ann.out_tar2.vld{1,3}],1);
            ypl{4,1} = mean([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2},ann.out_tar2.tst{1,3}],1);

            err{2,1}(:,1) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2},ann.out_tar2.vld{1,3}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2},ann.out_tar2.vld{1,3}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2},ann.out_tar2.tst{1,3}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2},ann.out_tar2.tst{1,3}],84,1);
        else
           ypl{2,1} = mean([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],1);
            ypl{3,1} = mean([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2},ann.out_tar.vld{1,3}],1);
            ypl{4,1} = mean([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2},ann.out_tar.tst{1,3}],1);

            err{2,1}(:,1) = prctile([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2},ann.out_tar.vld{1,3}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2},ann.out_tar.vld{1,3}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2},ann.out_tar.tst{1,3}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2},ann.out_tar.tst{1,3}],84,1);
        end
    elseif strcmp(component,'h12')
        if strcmp(TransferLearning,'True')
            ypl{2,1} = mean([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],1);
            ypl{3,1} = mean([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2}],1);
            ypl{4,1} = mean([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2}],1);

            err{2,1}(:,1) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}]-[ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}]-[ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}]-[ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2}],84,1);
        else
            ypl{2,1} = mean([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],1);
            ypl{3,1} = mean([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2}],1);
            ypl{4,1} = mean([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2}],1);

            err{2,1}(:,1) = prctile([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}]-[ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}]-[ann.out_tar.vld{1,1},ann.out_tar.vld{1,2}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}]-[ann.out_tar.tst{1,1},ann.out_tar.tst{1,2}],84,1);
        end
    else %not h12
        if strcmp(TransferLearning,'True')
            ypl{2,1} = mean([ann.out_trn2.trn{1,1}]-[ann.out_tar2.trn{1,1}],1);
            ypl{3,1} = mean([ann.out_trn2.vld{1,1}]-[ann.out_tar2.vld{1,1}],1);
            ypl{4,1} = mean([ann.out_trn2.tst{1,1}]-[ann.out_tar2.tst{1,1}],1);

            err{2,1}(:,1) = prctile([ann.out_trn2.trn{1,1}]-[ann.out_tar2.trn{1,1}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn2.trn{1,1}]-[ann.out_tar2.trn{1,1}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn2.vld{1,1}]-[ann.out_tar2.vld{1,1}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn2.vld{1,1}]-[ann.out_tar2.vld{1,1}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn2.tst{1,1}]-[ann.out_tar2.tst{1,1}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn2.tst{1,1}]-[ann.out_tar2.tst{1,1}],84,1);
        else
            ypl{2,1} = mean([ann.out_trn.trn{1,1}]-[ann.out_tar.trn{1,1}],1);
            ypl{3,1} = mean([ann.out_trn.vld{1,1}]-[ann.out_tar.vld{1,1}],1);
            ypl{4,1} = mean([ann.out_trn.tst{1,1}]-[ann.out_tar.tst{1,1}],1);

            err{2,1}(:,1) = prctile([ann.out_trn.trn{1,1}]-[ann.out_tar.trn{1,1}],16,1);
            err{2,1}(:,2) = prctile([ann.out_trn.trn{1,1}]-[ann.out_tar.trn{1,1}],84,1);

            err{3,1}(:,1) = prctile([ann.out_trn.vld{1,1}]-[ann.out_tar.vld{1,1}],16,1);
            err{3,1}(:,2) = prctile([ann.out_trn.vld{1,1}]-[ann.out_tar.vld{1,1}],84,1);

            err{4,1}(:,1) = prctile([ann.out_trn.tst{1,1}]-[ann.out_tar.tst{1,1}],16,1);
            err{4,1}(:,2) = prctile([ann.out_trn.tst{1,1}]-[ann.out_tar.tst{1,1}],84,1);
        end
    end
    
    figure('position',[0,0,13,10]);
    
    pl11=plot(xpl{2,1},ypl{2,1}(1:nT)); hold all;
%     pl11.LineWidth=4;
%     pl11.Color=rgb('lightgrey');
    pl11.LineStyle='none';
    pl21=bar(xpl{2,1}([1,3,5:nT]),err{2,1}([1,3,5:nT],1)); hold all;
    pl3=bar(xpl{2,1}([1,3,5:nT]),err{2,1}([1,3,5:nT],2)); hold all;
    pl21.BarWidth=0.9;
    pl3.BarWidth=0.9;
    
    pl21.FaceColor=rgb('lightgrey');
    pl3.FaceColor=rgb('lightgrey');
    
    pl22=plot(xpl{3,1},ypl{3,1}(1:nT)); hold all;
%     pl22.LineWidth=4;
%     pl22.Color=[0.4,0.4,0.4];
    pl22.LineStyle='none';
    pl23=bar(xpl{3,1}([1,3,5:nT]),err{3,1}([1,3,5:nT],1)); hold all;
    pl3=bar(xpl{3,1}([1,3,5:nT]),err{3,1}([1,3,5:nT],2)); hold all;
    pl23.BarWidth=0.5;
    pl3.BarWidth=0.5;
    pl23.FaceColor=[0.4,0.4,0.4];
    pl3.FaceColor=[0.4,0.4,0.4];
    
    pl33=plot(xpl{4,1},ypl{4,1}(1:nT)); hold all;
    pl33.LineStyle='none';
%     pl33.LineWidth=4;
%     pl33.Color=rgb('black');
    pl24=bar(xpl{4,1}([1,3,5:nT]),err{4,1}([1,3,5:nT],1)); hold all;
    pl3=bar(xpl{4,1}([1,3,5:nT]),err{4,1}([1,3,5:nT],2)); hold all;
    pl24.BarWidth=0.2;
    pl3.BarWidth=0.2;
    pl24.FaceColor=rgb('black');
    pl3.FaceColor=rgb('black');
    
    xlim(gca,xlm);
    ylim(gca,ylm);
    set(gca,'xtick',xtk,'ytick',ytk,'linewidth',2);
    set(gca,'ticklength',[.02,.02]);
    xlabel(gca,'T/T*','fontsize',15,'fontweight','bold');
    ylabel(gca,'log_{10}(Sa_{ANN}/Sa_{Obs})','fontsize',15,'fontweight','bold');
    leg=legend(gca,[pl21,pl23,pl24],{'TRN';'VLD';'TST'});
    
    set(leg,'interpreter','latex','location','northeast',...
        'orientation','horizontal','box','off');
    
    text(0.6,-0.5,strcat('$T^\star=$',num2str(TnC,'%.2f'),'$s$'),'parent',gca,...
        'interpreter','latex','fontsize',18)
    rule_fig(gcf);
    
    saveas(gcf,fullfile(wd,strcat(dsg.fnm,'_',['TRN',num2str(dsg.net.divideParam.trainRatio*100),'VAL',num2str(dsg.net.divideParam.valRatio*100),'TES',num2str(dsg.net.divideParam.testRatio*100)],dbn_name,'_',verNet)),'jpeg'); 
    close(gcf);
    return
end
