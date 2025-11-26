%% *Fancy Plotting*
% _Editor: Filippo Gatti
% CentraleSupélec - Laboratoire MSSMat
% DICA - Politecnico di Milano
% Copyright 2016_
%% NOTES
% _fpplot_: function plot fancy images from data
%% INPUT:
%
%% OUTPUT:
%
%% N.B.
%Need for _plot_set_up.m, format_figures.m, rule_fig.m_
function [varargout] = fpplot(varargin)
    
    %% SET-UP
    % _plot style_
    plot_set_up;
    warning off
    %%
    % _figure parameters_
    % figure position
    def.pfg = [0 0 16 16];
    def.vfg = 'off';
    %%
    % _axes parameters
    % plot dimension
    def.dim = 2;
    exp.dim = [2,3];
    % subplot grid
    def.spg = [1;1];
    def.spn = def.spg(1)*def.spg(2);
    % x-vector
    def.xpl = 1:10;
    % y-vector
    def.ypl = 1:10;
    % z-vector
    def.zpl = [];
    % parent axis
    def.pax = {1};
    % scale
    def.scl = {'lin'};
    exp.scl = ['lin','slx','sly','log'];
    % marker
    def.mrk = {'none'};
    def.lwd = 1.5;
    % x-label
    def.xlb = {''};
    % y-label
    def.ylb = {''};
    % title
    def.tit = {''};
    % legend
    def.leg = {''};
    % grid
    def.grd = {'off'};
    fxlm=0;fylm=0;fxtk=0;fytk=0;fpax=0;flst=0;flwd=0;
    % format_figure
    def.frf = {'s2a'};

    %%
    % _parser parameters_
    inp = inputParser;
    addParameter(inp,'pfg',def.pfg,@isnumeric);
    addParameter(inp,'nfg',@ischar);
    addParameter(inp,'vfg',def.vfg,@ischar);
    addParameter(inp,'sfg',@ischar);
    addParameter(inp,'dim',def.dim,@isnumeric);
    addParameter(inp,'spg',def.spg,@isnumeric);
    addParameter(inp,'xtk',@isnumeric);
    addParameter(inp,'ytk',@isnumeric);
    addParameter(inp,'ztk',@isnumeric);
    addParameter(inp,'xlm',@isnumeric);
    addParameter(inp,'ylm',@isnumeric);
    addParameter(inp,'zlm',@isnumeric);
    addParameter(inp,'scl',def.scl,@(x) any(~isempty(strfind(x,exp.scl))));
    addParameter(inp,'xlb',def.xlb);
    addParameter(inp,'ylb',def.ylb);
    addParameter(inp,'tit',def.tit);
    addParameter(inp,'leg',def.leg);
    addParameter(inp,'xpl',def.xpl,@iscell);
    addParameter(inp,'ypl',def.ypl,@iscell);
    addParameter(inp,'zpl',def.zpl,@iscell);
    addParameter(inp,'pax',def.pax,@iscell);
    addParameter(inp,'lst',@iscell);
    addParameter(inp,'lwd',def.lwd,@isnumeric);
    addParameter(inp,'mrk',def.mrk,@iscell);
    addParameter(inp,'grd',def.grd,@iscell);
    addParameter(inp,'frf',def.frf,@ischar);
    %% PARSE INPUTS
    % _parse input_
    
    parse(inp,varargin{:});
    
    % number of entities to be plotted
    spn = numel(inp.Results.xpl);
    try any(validatestring('vfg',inp.UsingDefaults));
        vfg = def.vfg;
    catch
        vfg = inp.Results.vfg;
    end
    %%
    % _input checkout_
    try any(validatestring('nfg',inp.UsingDefaults));
        hfg = figure('position',inp.Results.pfg,'visible',vfg);
    catch
        hfg = figure('name',inp.Results.nfg,'position',inp.Results.pfg,...
            'visible',vfg);
    end
    if spn>1
        %%
        % _subplot indexes_
        try any(validatestring('pax',inp.UsingDefaults));
            fpax=1;
            pax = num2cell(repmat(def.pax{1},spn,1),spn);
        catch
            pax = inp.Results.pax;
        end
        %%
        % _subplot grid_
        try any(validatestring('spg',inp.UsingDefaults));
            if fpax
                spg = def.spg;
            else
                spg(1) = floor(spn/2)+1;
                spg(2) = ceil(spn/2);
            end
        catch
            spg = inp.Results.spg;
        end
        
        %%
        % _x-limits_
        try any(validatestring('xlm',inp.UsingDefaults));
            fxlm = 0;
        catch
            xlm = inp.Results.xlm;
            fxlm = 1;
            if numel(xlm)==1
                xlm = repmat(xlm,spn,1);
            end
        end
        %%
        % _y-limits_
        try any(validatestring('ylm',inp.UsingDefaults));
            fylm=0;
        catch
            ylm = inp.Results.ylm;
            fylm = 1;
            if numel(ylm)==1
                ylm = repmat(ylm,spn,1);
            end
        end
        %%
        % _x-tick_
        try any(validatestring('xtk',inp.UsingDefaults));
        catch
            xtk = inp.Results.xtk;
            fxtk = 1;
            if numel(xtk)==1
                xtk = repmat(xtk,spn,1);
            end
        end
        %%
        % _y-tick_
        try any(validatestring('ytk',inp.UsingDefaults));
        catch
            ytk = inp.Results.ytk;
            fytk = 1;
            if numel(ytk)==1
                ytk = repmat(ytk,spn,1);
            end
        end
        %%
        % _grid_
        try any(validatestring('grd',inp.UsingDefaults));
            grd = num2cell(repmat(def.grd(1),spn,1));
        catch
            grd = repmat(inp.Results.grd(1),spn,1);
        end
        %%
        % _plot linestyle_
        try any(validatestring('lst',inp.UsingDefaults));
            
        catch
            flst=1;
            lst = inp.Results.lst;
            if numel(lst)==1
                lst = repmat(lst,spn,1);
            end
        end
        try any(validatestring('lwd',inp.UsingDefaults));
            
        catch
            flwd=1;
            lwd = inp.Results.lwd;
            if numel(lwd)==1
                lwd = repmat(lwd,spn,1);
            end
        end
        
        %%
        % _plot markers_
        try any(validatestring('mrk',inp.UsingDefaults));
            fmrk=1;
            mrk = num2cell(repmat(def.mrk{1},spn,1),spn);
        catch
            mrk = inp.Results.mrk;
        end
        if numel(mrk)==1
            mrk = repmat(mrk,spn,1);
        end
        %%
        % _x-label_
        try any(validatestring('xlb',inp.UsingDefaults));
            xlb = {''};
        catch
            xlb = inp.Results.xlb;
        end
        if numel(xlb)==1
            xlb = repmat(xlb,spn,1);
        end
        %%
        % _y-label_
        try any(validatestring('ylb',inp.UsingDefaults));
            ylb = {''};
        catch
            ylb = inp.Results.ylb;
        end
        if numel(ylb)==1
            ylb = repmat(ylb,spn,1);
        end
        %%
        % _title_
        try any(validatestring('tit',inp.UsingDefaults));
            tit = {''};
        catch
            tit = inp.Results.tit;
        end
        if numel(tit)==1
            tit = repmat(tit,spn,1);
        end
        %%
        % _subplot indexes_
        try any(validatestring('scl',inp.UsingDefaults));
            scl = repmat(def.scl(1),spn,1);
        catch
            scl = inp.Results.scl;
        end
    else
        %%
        % _default parameters_
        spn = def.spn;
        spg = def.spg;
        pax = def.pax;
        scl = def.scl;
        %%
        % _x-limits_
        try any(validatestring('xlm',inp.UsingDefaults));
            fxlm=0;
        catch
            xlm = inp.Results.xlm;
            fxlm = 1;
        end
        %%
        % _y-limits_
        try any(validatestring('ylm',inp.UsingDefaults));
            fylm=0;
        catch
            ylm = inp.Results.ylm;
            fylm = 1;
        end
        %%
        % _x-tick_
        try any(validatestring('xtk',inp.UsingDefaults));
        catch
            xtk = inp.Results.xtk;
            fxtk = 1;
        end
        %%
        % _y-tick_
        try any(validatestring('ytk',inp.UsingDefaults));
        catch
            ytk = inp.Results.ytk;
            fytk = 1;
        end
        %%
        try any(validatestring('xlb',inp.UsingDefaults));
            xlb = def.xlb;
        catch
            xlb = inp.Results.xlb;
        end
        try any(validatestring('ylb',inp.UsingDefaults));
            ylb = def.ylb;
        catch
            ylb = inp.Results.ylb;
        end
        try any(validatestring('tit',inp.UsingDefaults));
            tit = def.tit;
        catch
            tit = inp.Results.tit;
        end
        try any(validatestring('grd',inp.UsingDefaults));
            grd = def.grd;
        catch
            grd = inp.Results.grd;
        end
        
        try any(validatestring('mrk',inp.UsingDefaults));
            mrk = def.mrk;
        catch
            mrk = inp.Results.mrk;
        end
        try any(validatestring('lst',inp.UsingDefaults));
        catch
            lst = inp.Results.lst;
            flst = 1;
        end
        try any(validatestring('lwd',inp.UsingDefaults));
        catch
            lwd = inp.Results.lwd;
            flwd = 1;
        end
        %%
        % _subplot indexes_
        try any(validatestring('scl',inp.UsingDefaults));
        catch
            scl = inp.Results.scl;
        end
        
    end
    
    %% PLOT FIGURE
    flg = zeros(spn,1);
    paxt = [];% cell2mat(pax);
    n_ = 0;
    flag_modify=-ones(spn,1);
    for m_ = 1:spn
        %%
        % _plot_
        %if ismember(pax{m_},paxt(1:m_-1))
            % idx = find(paxt(1:m_-1)==pax{m_},1,'first');
        if ismember(pax{m_},paxt)
            cntt(n_) = cntt(n_)+1;
            idx = find(paxt==pax{m_},1,'first');
            haxt = hax(idx);
            if flst
                hpl(idx,cntt(n_))=plot(haxt,inp.Results.xpl{m_},inp.Results.ypl{m_},...
                    'marker',mrk{m_},'linestyle',lst{m_});
            else
                hpl(idx,cntt(n_))=plot(haxt,inp.Results.xpl{m_},inp.Results.ypl{m_},...
                    'marker',mrk{m_});
            end
            if flwd 
                hpl(idx,cntt(n_)).LineWidth = lwd(m_);
            end
            hpl(idx,cntt(n_)).MarkerFaceColor = hpl(idx,cntt(n_)).Color;
            hpl(idx,cntt(n_)).MarkerEdgeColor = [0,0,0];
        else
            n_=n_+1;
            cntt(n_) = 1;
            paxt = [paxt;pax{m_}];
            haxt = subplot(spg(1),spg(2),pax{m_},'parent',hfg);
            hold(haxt,'all');
            hax(n_) = haxt;
            if flst
                hpl(n_,cntt(n_))=plot(haxt,inp.Results.xpl{m_},inp.Results.ypl{m_},...
                    'marker',mrk{m_},'linestyle',lst{m_});
            else
                hpl(n_,cntt(n_))=plot(haxt,inp.Results.xpl{m_},inp.Results.ypl{m_},...
                    'marker',mrk{m_}');
            end
            if flwd 
                hpl(n_,cntt(n_)).LineWidth = lwd(m_);
            end
            hpl(n_,cntt(n_)).MarkerFaceColor = hpl(n_,cntt(n_)).Color;
            hpl(n_,cntt(n_)).MarkerEdgeColor = [0,0,0];
            flag_modify(n_) = 1;
        end
        
    end
    %% AXES SET UP
    for mm_ = 1:numel(paxt)
        if logical(flag_modify(mm_))
            m_ = paxt(mm_);
            %%
            % _legend_
            idx = ~strcmpi(inp.Results.leg{m_},'');
            
            if any(idx)
                legg=legend(hax(m_),hpl(m_,idx),inp.Results.leg{m_}(idx));
                set(legg,'interpreter','latex','box','off');
            end
            %%
            % _axes labels_
            xlabel(hax(m_),xlb{m_});
            ylabel(hax(m_),ylb{m_});
            %%
            % _xlim_
            if fxlm
                xlim(hax(m_),xlm{m_});
                if ~fxtk
                    xtkt=get(hax(m_),'xtick'); xtkt=xtkt(:);
                    if xlm{m_}(1)<xtkt
                        xtkt=[xlm{m_}(1);xtkt];
                    end
                    if xlm{m_}(end)>xtkt
                        xtkt=[xtkt;xlm{m_}(end)];
                    end
                    set(hax(m_),'xtick',xtkt);
                end
            else
                if fxtk
                    xlim(hax(m_),[xtk{m_}(1),xtk{m_}(end)]);
                else
                    [xlm,~]=get_axis_lim(inp.Results.xpl,inp.Results.ypl,1,1);
                    if all(~isnan(xlm))
                        xlim(hax(m_),xlm);
                    end
                end
            end
            %%
            % _ylim_
            if fylm
                ylim(hax(m_),ylm{m_});
                if ~fxtk
                    ytkt=get(hax(m_),'xtick'); ytkt=ytkt(:);
                    if ylm{m_}(1)<ytkt
                        ytkt=[ylm{m_}(1);ytkt];
                    end
                    if ylm{m_}(end)>ytkt
                        ytkt=[ytkt;ylm{m_}(end)];
                    end
                    set(hax(m_),'ytick',ytkt);
                end
            else
                if fytk
                    ylim(hax(m_),[ytk{m_}(1),ytk{m_}(end)]);
                else
                    [~,ylm]=get_axis_lim(inp.Results.xpl,inp.Results.ypl,1,1);
                    if all(~isnan(ylm))
                        ylim(hax(m_),ylm);
                    end
                end
            end
            %%
            % _xtick_
            if fxtk
                set(hax(m_),'xtick',xtk{m_});
            end
            %%
            % _ytick_
            if fytk
                set(hax(m_),'ytick',ytk{m_});
            end
            % _axes grid
            if strcmpi(grd,'cool')
                set_axis_grid(hax(m_));
            else
                grid(hax(m_),grd{m_});
            end
            %%
            % _title_
            title(hax(m_),tit{m_});
            %%
            % _axes scale_
            switch scl{m_}
                case 'lin'
                    set(hax(m_),'xscale','lin','yscale','lin');
                case 'slx'
                    set(hax(m_),'xscale','log','yscale','lin');
                case 'sly'
                    set(hax(m_),'xscale','lin','yscale','log');
                case 'log'
                    set(hax(m_),'xscale','log','yscale','log');
            end
            %%
            % _format axes_
            if strcmpi(inp.Results.frf,'s2a')
                syn2ann_format_figures(hax(m_));
            else
                format_figures(hax(m_));
            end
        end
    end
    %%
    % _crop figure_
    rule_fig(hfg);

    %% *OUTPUT*
    varargout{1} = hfg;
    varargout{2} = hax;
    varargout{3} = hpl;
    return
end
