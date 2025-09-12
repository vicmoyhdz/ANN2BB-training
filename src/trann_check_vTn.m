function [varargout] = trann_check_vTn(varargin)
    
    inp = varargin{1};
    tar = varargin{2};
    chk = varargin{3};
    tol = varargin{4};

    % check input natural periods
    inp.idx = -999*ones(inp.nT,1);
    for i_ = 1:inp.nT
        inp.idx(i_,1) = find(abs(chk.vTn-inp.vTn(i_))<tol);
    end
    
    % check target natural periods
    tar.idx = -999*ones(tar.nT,1);
    for i_ = 1:tar.nT
        tar.idx(i_,1) = find(abs(chk.vTn-tar.vTn(i_))<tol);
    end
    if any(isempty(inp.idx))||any(isempty(tar.idx))
        error('no period found!');
    end
    varargout{1} = inp.idx;
    varargout{2} = tar.idx;

    return
end
