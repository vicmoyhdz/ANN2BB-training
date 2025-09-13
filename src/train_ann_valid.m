function [varargout] = train_ann_valid(varargin)
    %% *SET-UP*
    ann = varargin{1}; 
    TransferLearning = varargin{2}; 
    index_extra = varargin{3}; 
    n_classes = varargin{4}; 
    component= varargin{5}; 
            
            %% *ANN VALIDATION*
            fprintf('VALIDATING...\n');
 if strcmp(component,'h12v')
            if index_extra>0 && n_classes>0
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4},ann.inp.trn{1,5});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4},ann.inp.vld{1,5});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4},ann.inp.tst{1,5});
            elseif index_extra>0 && n_classes==0
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4});
            else
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3});
            end

    if strcmp(TransferLearning,'True')
         if index_extra>0 && n_classes>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4},ann.inp2.trn{1,5});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann2.inp.vld{1,4},ann.inp2.vld{1,5});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4},ann.inp2.tst{1,5});
            elseif index_extra>0 && n_classes==0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4});
            else
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3});
            end
    end
elseif strcmp(component,'h12')
            if index_extra>0 && n_classes>0
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3},ann.inp.trn{1,4});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3},ann.inp.vld{1,4});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3},ann.inp.tst{1,4});
            elseif index_extra>0 && n_classes==0
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3});
            else
            [ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2});
            [ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2});
            [ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2});
            end

    if strcmp(TransferLearning,'True')

     if index_extra>0 && n_classes>0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3},ann.inp2.trn{1,4});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3},ann.inp2.vld{1,4});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3},ann.inp2.tst{1,4});
     elseif index_extra>0 && n_classes==0
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3});
     else
            [ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2});
            [ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2});
            [ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2});
     end
    end
 else %not h12

            if index_extra>0 && n_classes>0
            [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2},ann.inp.trn{1,3});
            [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2},ann.inp.vld{1,3});
            [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2},ann.inp.tst{1,3});
            elseif index_extra>0 && n_classes==0
            [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1},ann.inp.trn{1,2});
            [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1},ann.inp.vld{1,2});
            [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1},ann.inp.tst{1,2});
            else
            [ann.out_trn.trn{1,1}] = predict(ann.net,ann.inp.trn{1,1});
            [ann.out_trn.vld{1,1}] = predict(ann.net,ann.inp.vld{1,1});
            [ann.out_trn.tst{1,1}] = predict(ann.net,ann.inp.tst{1,1});
            end

    if strcmp(TransferLearning,'True')

     if index_extra>0 && n_classes>0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2},ann.inp2.trn{1,3});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2},ann.inp2.vld{1,3});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2},ann.inp2.tst{1,3});
     elseif index_extra>0 && n_classes==0
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1},ann.inp2.trn{1,2});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1},ann.inp2.vld{1,2});
            [ann.out_trn2.tst{1,1}] = predict(ann.net,ann.inp2.tst{1,1},ann.inp2.tst{1,2});
     else
            [ann.out_trn2.trn{1,1}] = predict(ann.net,ann.inp2.trn{1,1});
            [ann.out_trn2.vld{1,1}] = predict(ann.net,ann.inp2.vld{1,1});
            [ann.out_trn2.tst{1,1},] = predict(ann.net,ann.inp2.tst{1,1});
     end
    end

 end
            
            %% *DEFINE INPUTS/TARGETS SUBSETS FROM TRAIN PROCEDURE*
            
            % _TRANING/VALIDATION/TEST SUBSET VALUES_
            ann.out_tar.trn = ann.tar.trn;
            ann.out_tar.vld = ann.tar.vld;
            ann.out_tar.tst = ann.tar.tst;

if strcmp(TransferLearning,'True')
            ann.out_tar2.trn = ann.tar2.trn;
            ann.out_tar2.vld = ann.tar2.vld;
            ann.out_tar2.tst = ann.tar2.tst;
end

            %% *COMPUTE PERFORMANCE*
            fprintf('COMPUTING PERFORMANCE...\n')

if strcmp(component,'h12v')
if strcmp(TransferLearning,'True')
       ann.prf.trn = mse([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],[ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}]);
       ann.prf.vld = mse([ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2},ann.out_tar2.vld{1,3}],[ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2},ann.out_trn2.vld{1,3}]);
       ann.prf.tst = mse([ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2},ann.out_tar2.tst{1,3}],[ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2},ann.out_trn2.tst{1,3}]);
       ann.prf.r = (regression(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],1,[]),reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2},ann.out_tar2.trn{1,3}],1,[])-reshape([ann.out_trn2.trn{1,1},...
           ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[])))/length(reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2},ann.out_trn2.trn{1,3}],1,[]));
else
       ann.prf.trn = mse([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],[ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}]);
       ann.prf.vld = mse([ann.out_tar.vld{1,1},ann.out_tar.vld{1,2},ann.out_tar.vld{1,3}],[ann.out_trn.vld{1,1},ann.out_trn.vld{1,2},ann.out_trn.vld{1,3}]);
       ann.prf.tst = mse([ann.out_tar.tst{1,1},ann.out_tar.tst{1,2},ann.out_tar.tst{1,3}],[ann.out_trn.tst{1,1},ann.out_trn.tst{1,2},ann.out_trn.tst{1,3}]);
       ann.prf.r = (regression(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],1,[]),reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2},ann.out_tar.trn{1,3}],1,[])-reshape([ann.out_trn.trn{1,1},...
           ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[])))/length(reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2},ann.out_trn.trn{1,3}],1,[]));
end   
elseif strcmp(component,'h12')
if strcmp(TransferLearning,'True')
       ann.prf.trn = mse([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],[ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}]);
       ann.prf.vld = mse([ann.out_tar2.vld{1,1},ann.out_tar2.vld{1,2}],[ann.out_trn2.vld{1,1},ann.out_trn2.vld{1,2}]);
       ann.prf.tst = mse([ann.out_tar2.tst{1,1},ann.out_tar2.tst{1,2}],[ann.out_trn2.tst{1,1},ann.out_trn2.tst{1,2}]);
       ann.prf.r = (regression(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],1,[]),reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar2.trn{1,1},ann.out_tar2.trn{1,2}],1,[])-reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}],1,[])))/length(reshape([ann.out_trn2.trn{1,1},ann.out_trn2.trn{1,2}],1,[]));
else
       ann.prf.trn = mse([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],[ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}]);
       ann.prf.vld = mse([ann.out_tar.vld{1,1},ann.out_tar.vld{1,2}],[ann.out_trn.vld{1,1},ann.out_trn.vld{1,2}]);
       ann.prf.tst = mse([ann.out_tar.tst{1,1},ann.out_tar.tst{1,2}],[ann.out_trn.tst{1,1},ann.out_trn.tst{1,2}]);
       ann.prf.r = (regression(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],1,[]),reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar.trn{1,1},ann.out_tar.trn{1,2}],1,[])-reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}],1,[])))/length(reshape([ann.out_trn.trn{1,1},ann.out_trn.trn{1,2}],1,[]));
end
else %not h12
    if strcmp(TransferLearning,'True')
       ann.prf.trn = mse([ann.out_tar2.trn{1,1}],[ann.out_trn2.trn{1,1}]);
       ann.prf.vld = mse([ann.out_tar2.vld{1,1}],[ann.out_trn2.vld{1,1}]);
       ann.prf.tst = mse([ann.out_tar2.tst{1,1}],[ann.out_trn2.tst{1,1}]);
       ann.prf.r = (regression(reshape([ann.out_tar2.trn{1,1}],1,[]),reshape([ann.out_trn2.trn{1,1}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar2.trn{1,1}],1,[])-reshape([ann.out_trn2.trn{1,1}],1,[])))/length(reshape([ann.out_trn2.trn{1,1}],1,[]));
else
       ann.prf.trn = mse([ann.out_tar.trn{1,1}],[ann.out_trn.trn{1,1}]);
       ann.prf.vld = mse([ann.out_tar.vld{1,1}],[ann.out_trn.vld{1,1}]);
       ann.prf.tst = mse([ann.out_tar.tst{1,1}],[ann.out_trn.tst{1,1}]);
       ann.prf.r = (regression(reshape([ann.out_tar.trn{1,1}],1,[]),reshape([ann.out_trn.trn{1,1}],1,[])))^2;
       ann.prf.mae = sum(abs(reshape([ann.out_tar.trn{1,1}],1,[])-reshape([ann.out_trn.trn{1,1}],1,[])))/length(reshape([ann.out_trn.trn{1,1}],1,[]));
end

end
    
    %% *OUTPUT*
    varargout{1} = ann;
    return
end
