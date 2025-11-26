function rule_fig(fig)
    
    hax = get(fig,'children'); 
    leg = findobj(hax,'type','legend');
    %set(leg,'box','off','fontweight','bold','fontsize',15,'interpreter','none');
    hax = findobj(hax,'type','axes');
    ti_all  = get(hax,'tightinset');
    pos_all = get(hax,'position');
    if numel(hax)==1
        hax = [hax];
        ti_all = {ti_all};
        pos_all = {pos_all};
    end
    pos_lim = [1 1 0 0];
    ti_lim  = [0 0];
    % position and inset limits
    
    for i = 1: numel(hax)
        if pos_all{i}(1) < pos_lim(1)
            pos_lim(1) = pos_all{i}(1);
        end
        if pos_all{i}(2) < pos_lim(2)
            pos_lim(2) = pos_all{i}(2);
        end
        if pos_all{i}(3) > pos_lim(3)
            pos_lim(3) = pos_all{i}(3);
        end
        if pos_all{i}(4) > pos_lim(4)
            pos_lim(4) = pos_all{i}(4);
        end
    end
    for i = 1: numel(hax)
        if ti_all{i}(1) > ti_lim(1)
            ti_lim(1) = ti_all{i}(1);
        end
        if ti_all{i}(2) > ti_lim(2)
            ti_lim(2) = ti_all{i}(2);
        end
    end
    % redefine axes positions    
    
    for i = 1:numel(hax)
        pos_loc = get(hax(i),'position');
        
        set(hax(i),'position',...
            [ti_lim(1)-pos_lim(1)+pos_loc(1), ...
            ti_lim(2)-pos_lim(2)+pos_loc(2), ...
            pos_lim(3), ...
            pos_loc(4)]);
    end
    
    set(fig,'units','centimeters','paperunits','centimeters');
    set(hax,'units','centimeters');
    
    pos_all = get(hax,'position');
    ti_all  = get(hax,'tightinset');
    
    ti_lim  = zeros(1,2); pos=zeros(1,2);
    % position and inset limits
    if numel(hax)==1
        pos_all = {pos_all};
        ti_all  = {ti_all};
    end
    
    for i = 1: numel(hax)
        if ti_all{i}(1) > ti_lim(1)
            ti_lim(1) = ti_all{i}(1);
        end
        if ti_all{i}(2) > ti_lim(2)
            ti_lim(2) = ti_all{i}(2);
        end
        
    end
    
    for i = 1:numel(hax)
        
        pos_loc_x = pos_all{i}(1)+pos_all{i}(3)+ti_all{i}(3);
        pos(1)  = max(pos_loc_x,pos(1));
        pos_loc_y = pos_all{i}(2)+pos_all{i}(4)+ti_all{i}(4);
        pos(2)  = max(pos_loc_y,pos(2));
        
    end
    
    set(fig,'position',[0 0 pos]);
    
    set(fig,'papersize',pos);
    set(fig,'paperpositionmode','manual');
    set(fig,'paperposition',[0 0 pos]);
    
    return
end