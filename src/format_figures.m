%==== format figures


function format_figures(hax)
    
    if nargin==0
        hax=gca;
    end
    
    set(hax,'box','on','xminortick','on','yminortick','on');
    %
    set(hax,'ticklength',[.02,.02]);
    ti = get(hax,'title');
    set(ti,'fontsize',25,'fontweight','bold');%30
    xl = get(hax,'xlabel');
    set(xl,'fontsize',20,'fontweight','bold');%24
    yl = get(hax,'ylabel');
    set(yl,'fontsize',20,'fontweight','bold');%24
    zl = get(hax,'zlabel');
    set(zl,'fontsize',20);%24
    li = get(hax,'children');
    set(li,'linewidth',2.5,'markersize',5);%3
    set(hax,'linewidth',2.2,'fontsize',15);   %3,20
   
end