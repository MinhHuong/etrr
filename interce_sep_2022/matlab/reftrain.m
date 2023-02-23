function data_av=reftrain(filename,data_av)

splitname=strsplit(filename,{'_'});

if isfield(data_av,'ctxt')==0
    ctxt=table();
    data_av.ctxt=ctxt;
end
    
%% train
train=splitname{1};
data_av.ctxt.train_sncf=str2double(train(2:end));
data_av.ctxt.train=(str2double(train(end-3:end))+1)/2;

%% vehicule
vehicule=splitname{4};
data_av.ctxt.vehicule_sncf=str2double([splitname{3},splitname{4}]);
vehiculetemp=str2double(vehicule(1:2));
switch vehiculetemp
    case 0
        data_av.ctxt.vehicule=11;
    case 14
        data_av.ctxt.vehicule=12;
    case 21
        data_av.ctxt.vehicule=13;
    case 54
        data_av.ctxt.vehicule=14;
    case 62
        data_av.ctxt.vehicule=15;
    case 74
        data_av.ctxt.vehicule=16;
    case 83
        data_av.ctxt.vehicule=17;
    case 07
        data_av.ctxt.vehicule=20;
end

        

%% DCU
dcu=splitname{5};
data_av.ctxt.dcu=str2double(dcu(end));














end %function