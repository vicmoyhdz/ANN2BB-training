function [varargout]=plot_set_up
    
    %% GLOBAL VARIABLES
    %_ticks/limits for fourier spectra_
    xt_freq   = [.1,1,10,25]; 
    xl_freq   = [xt_freq(1),xt_freq(end)];
    
    %%
    %_ticks/limits for response spectra_
    xt_period = [1e-2,1e-1,1,5,10];  
    xl_period = [xt_period(1),xt_period(end)];
    
    %% DEFAULT UNITS
    set(0,'defaultuicontrolunits','centimeters');
    set(0,'defaultfigureunits','centimeters');
    
    %% DEFAULT POSITIONS
    fig_position=[0,0,10,10];
    hax_position=[.15 .15 .8 .775];    
    set(0,'defaultfigureposition',fig_position);    
    set(0,'defaultaxesactivepositionproperty','position');
    set(0,'defaultaxesposition',hax_position);
    
    %% DEFAULT PLOT STYLE
    set(0,'defaultaxesfontsize',20);
    set(0,'defaultaxesxminortick','on','defaultaxesyminortick','on');
%     %% DEFAULT PLOT COLORS
%     color_order = [0.000 0.000 1.000;
%                    1.000 0.000 0.000;
%                    0.000 1.000 0.000;
%                    0.000 0.000 0.000;
%                    0.929 0.694 0.125;
%                    0.600 0.600 0.600;
%                    0.301 0.745 0.933;
%                    0.200 0.400 1.000;
%                    0.400 0.000 0.600;
%                    0.000 0.447 0.741;
%                    1.000 0.500 0.200;
%                    0.850 0.325 0.098;
%                    0.635 0.078 0.184;
%                    0.494 0.184 0.556;
%                    0.466 0.674 0.188;
%                    0.120 0.600 0.000;
%                    1.000 0.498 0.140;
%                    0.800 0.060 0.460];
%     color_order=[color_order;varycolor(100)];    
%     set(0,'defaultaxescolororder',color_order);
    %% DEFAULT GUI APPEARANCE
    set(0,'defaultuicontrolbackgroundcolor',[0.94 0.94 0.94]);
    set(0,'defaultuicontrolbusyaction','cancel');% So multiple pushes don't stack.
    set(0,'defaultuicontrolinterrupt','off');
    set(0,'defaultuicontrolfontweight','bold');
    set(0,'defaultuicontrolfontsize',14);
    
    %% OUTPUT
    varargout{1} = xt_freq;
    varargout{2} = xl_freq;
    varargout{3} = xt_period;
    varargout{4} = xl_period;
    return
end
