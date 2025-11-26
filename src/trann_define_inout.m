

function [varargout] = trann_define_inout(varargin)
    
    TnC = varargin{1};
    all_periods=[0,0.04,0.05,0.07,(0.1:0.05:0.5),0.6,0.7,0.75,0.8,0.9,1.0:0.2:2.0,(2.5:0.5:5)];
    inp.vTn = all_periods(all_periods>=TnC);
    tar.vTn = all_periods(all_periods<TnC);
    
    inp.nT = length(inp.vTn);
    tar.nT = length(tar.vTn);
    
    varargout{1} = inp.vTn(:);
    varargout{2} = tar.vTn(:);
    varargout{3} = inp.nT;
    varargout{4} = tar.nT;
    
end