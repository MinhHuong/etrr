function [data_av,cycle_por,cycle_mm,marqueur,error]=preprocess(data_av,filename,error)
try
    %% init
    cycle_por=struct();
    cycle_por.marqueur=table();
    cycle_mm=struct();
    cycle_mm.marqueurmm=table();
    marqueur=table();
    marqueurmm=table();
    
    
    if isfield(data_av,'ctxt')==0
        error.nb=2; %% cycle vide
        error.type='PREPROC';
        error.label='fichier sans context';
        data_av=reftrain(filename,data_av);
        data_av.ctxt.filename=filename;
        data_av.ctxt.mvt_mm=0;
        data_av.ctxt.mvt_por=0;
        return
    end
    
    if width(data_av.ctxt)<18
        error.nb=5; %% cycle vide
        error.type='PREPROC';
        error.label='fichier context imcomplet';
        data_av=reftrain(filename,data_av);
        data_av.ctxt.filename=filename;
        data_av.ctxt.mvt_mm=0;
        data_av.ctxt.mvt_por=0;
        return
    end
    
    data_av=reftrain(filename,data_av);
    data_av.ctxt.filename=filename;
    data_av.ctxt.mvt_mm=0;
    data_av.ctxt.mvt_por=0;
    
    if isfield(data_av,'courbe')==0
        error.nb=3; %% cycle vide
        error.type='PREPROC';
        error.label='fichier sans courbe';
        return
    end
    
    if sum(data_av.courbe.PMM,'omitnan')==0 && sum(data_av.courbe.PPOR,'omitnan')==0
        error.nb=4; %% cycle vide
        error.type='PREPROC';
        error.label='cycle vide';
        return
    end
    
    %% Decoupe ouverture/fermeture
    varana=timetable(data_av.courbe.Time,gradient(data_av.courbe.LT_AO1),'VariableNames',{'testltao1'});
    varana.testltao2=gradient(data_av.courbe.LT_AO2);
    varana.testltcf1=gradient(data_av.courbe.LT_CF);
    varana.testltcf2=gradient(data_av.courbe.LT_CF2);
    %% Preproc por
    if sum(data_av.courbe.PPOR,'omitnan')==0
        data_av.ctxt.mvt_por=0;
    else
        data_av.ctxt.mvt_por=1;
        data_av.courbe=sortrows(data_av.courbe,'Time','ascend');
        %% Marqueur de decoupage porte
        i_por=timetable(data_av.courbe.Time,data_av.courbe.IPOR,'VariableNames',{'I'});
        p_por=timetable(data_av.courbe.Time,data_av.courbe.PPOR,'VariableNames',{'P'});
        t_por=timetable(data_av.courbe.Time,data_av.courbe.TPOR,'VariableNames',{'T'});
        i_por2=fillmissing(i_por,'nearest');
        p_por2=fillmissing(p_por,'nearest');
        t_por2=fillmissing(t_por,'nearest');
        i_por.GI=gradient(i_por2.I);
        p_por.GP=gradient(p_por2.P);
        t_por.GT=gradient(t_por2.T);
        DI=zeros(length(i_por2.I),1);
        DI(2:end)=diff(i_por2.I);
        i_por.DI=DI;
        DP=zeros(length(p_por2.P),1);
        DP(2:end)=diff(p_por2.P);
        p_por.DP=DP;
        DT=zeros(length(t_por2.T),1);
        DT(2:end)=diff(t_por2.T);
        t_por.DT=DT;
        varana.varI=abs(i_por.DI);
        varana.varP=abs(p_por.DP)*10;
        varana.varT=abs(t_por.DT);
        varana.var=abs(i_por.DI)*5+abs(p_por.DP)*10+abs(t_por.DT);
        
        %% Variation booléenne
        varana.varbool(1,:)=0;
        varbl=varana.var>50;
        varana.varbool(varbl)=1;
        varana.varbool=boolfilter(varana.varbool);
        
        %% Nombre de cycle du fichier
        diffvar=diff(varana.varbool);
        marqueurstart=find(diffvar==1);
        marqueurstop=find(diffvar==-1);
        
        %% premier cycle complet ?
        minstart=min(marqueurstart);
        minstop=min(marqueurstop);
        if minstart<minstop
            pcc=1;
        else
            pcc=0;
        end
        
        %% dernier cycle complet ?
        maxstart=max(marqueurstart);
        maxstop=max(marqueurstop);
        if maxstart<maxstop
            dcc=1;
        else
            dcc=0;
        end
        
        %% nombre de cycle
        if pcc==1 && dcc==1
            nbcycle=length(marqueurstart);
        elseif pcc==0 && dcc==0
            nbcycle=length(marqueurstart)-2;
        else
            nbcycle=length(marqueurstart)-1;
        end
        
        %% marqueur cycle
        marqueur=table();
        for i=1:nbcycle
            marqueuradd=table();
            if pcc==1
                marqueuradd.cycle=i;
                if varana.Time(marqueurstart(i,1)+1)-varana.Time(marqueurstart(i,1))>milliseconds(100)
                    marqueuradd.mstart=marqueurstart(i,1)+1;
                else
                    marqueuradd.mstart=marqueurstart(i,1);
                end
                marqueuradd.mend=marqueurstop(i,1);
            else
                marqueuradd.cycle=i;
                if varana.Time(marqueurstart(i,1)+1)-varana.Time(marqueurstart(i,1))>milliseconds(100)
                    marqueuradd.mstart=marqueurstart(i+1,1)+1;
                else
                    marqueuradd.mstart=marqueurstart(i+1,1);
                end
                marqueuradd.mend=marqueurstop(i+1,1);
            end %pcc=1
            marqueur=[marqueur;marqueuradd];
        end %i=1:nbcycle
        
        %% type de cycle porte
        cycle.marqueur=marqueur;
        if ~isempty(marqueur)
            data_av.ctxt.nb_cycle=height(marqueur);
            % Cycle d'ouverture avec fermeture temporisé
            for i=1:height(marqueur)
                datacycle=data_av.courbe(marqueur.mstart(i,1):marqueur.mend(i,1),:);
                datacycle=datacycle(datacycle.SleepMode==0,:); % modification des marqueur ???? a faire plus tard
                ctxtcycle=table;
                if isempty(datacycle)
                    marqueur.sys{i,1}='P';
                    marqueur.type{i,1}='E';
                    ctxtcycle.id_type=6;
                    ctxtcycle.typelabel='Cycle Vide';
                    marqueur.tps(i,1)=duration(0,0,0);
                    ctxtcycle.temps_cycle=0;
                else
                    if abs(datacycle.PPOR(1,1)-datacycle.PPOR(end,1))<200
                        marqueur.sys{i,1}='P';
                        marqueur.type{i,1}='E';
                        ctxtcycle.id_type=4;
                        ctxtcycle.typelabel='Error deltaP=0';
                    else
                        if nanmax(datacycle.IPOR)<200
                            marqueur.sys{i,1}='P';
                            marqueur.type{i,1}='E';
                            ctxtcycle.id_type=5;
                            ctxtcycle.typelabel='Error bruit capteur I';
                        else
                            if nanmean(datacycle.TPOR)>0 %ouverture
                                marqueur.sys{i,1}='P';
                                marqueur.type{i,1}='O';
                                ctxtcycle.id_type=1;
                                ctxtcycle.typelabel='ouverture';
                            else %fermeture
                                if sum(datacycle.LT_CF)+sum(datacycle.LT_CF2)>0 %fermeture normale
                                    marqueur.sys{i,1}='P';
                                    marqueur.type{i,1}='F';
                                    ctxtcycle.id_type=2;
                                    ctxtcycle.typelabel='fermeture';
                                else %fermeture lente
                                    marqueur.sys{i,1}='P';
                                    marqueur.type{i,1}='FL';
                                    ctxtcycle.id_type=3;
                                    ctxtcycle.typelabel='fermeture lente';
                                end
                            end
                        end
                    end
                    marqueur.tps(i,1)=duration(datacycle.Time(end)-datacycle.Time(1));
                    ctxtcycle.temps_cycle=seconds(duration(datacycle.Time(end)-datacycle.Time(1)));
                end
                cycle_por.(['cycle_',num2str(i),'_data'])=datacycle;
                cycle_por.(['cycle_',num2str(i),'_ctxt'])=ctxtcycle;
                cycle_por.marqueur=marqueur;
            end %i=1:height(marqueur)
            %% error donnée de cycle incomplète ?
            %marqueurtracer(data_av,ConfigCBM,marqueur)
        else
            data_av.ctxt.nb_cycle=0;
            cycle_por.marqueur=table();
        end
    end
    
    %% preproc marche
    if sum(data_av.courbe.PMM,'omitnan')==0
        data_av.ctxt.mvt_mm=0;
        cycle_mm.marqueurmm=table();
    else
        data_av.ctxt.mvt_mm=1;
        
        %% marqueur de decoupage marche
        varanamm=timetable(data_av.courbe.Time,gradient(data_av.courbe.LT_AO1),'VariableNames',{'testltao1'});
        varanamm.testltao2=gradient(data_av.courbe.LT_AO2);
        varanamm.testltcf1=gradient(data_av.courbe.LT_CF);
        varanamm.testltcf2=gradient(data_av.courbe.LT_CF2);
        i_mm=timetable(data_av.courbe.Time,data_av.courbe.IMM,'VariableNames',{'I'});
        p_mm=timetable(data_av.courbe.Time,data_av.courbe.PMM,'VariableNames',{'P'});
        t_mm=timetable(data_av.courbe.Time,data_av.courbe.TMM,'VariableNames',{'T'});
        i_mm2=fillmissing(i_mm,'nearest');
        p_mm2=fillmissing(p_mm,'nearest');
        t_mm2=fillmissing(t_mm,'nearest');
        i_mm.GI=gradient(i_mm2.I);
        p_mm.GP=gradient(p_mm2.P);
        t_mm.GT=gradient(t_mm2.T);
        DI=zeros(length(i_mm2.I),1);
        DI(2:end)=diff(i_mm2.I);
        i_mm.DI=DI;
        DP=zeros(length(p_mm2.P),1);
        DP(2:end)=diff(p_mm2.P);
        p_mm.DP=DP;
        DT=zeros(length(t_mm2.T),1);
        DT(2:end)=diff(t_mm2.T);
        t_mm.DT=DT;
        varanamm.varmm=abs(i_mm.DI)*5+abs(p_mm.DP)*10+abs(t_mm.DT);
        
        %% Variation booléenne marche
        varanamm.varboolmm(1,:)=0;
        varbl=varanamm.varmm>75;
        varanamm.varboolmm(varbl)=1;
        varanamm.varboolmm=boolfilter(varanamm.varboolmm);
        
        %% Nombre de cycle du fichier marche
        diffvarmm=diff(varanamm.varboolmm);
        marqueurstartmm=find(diffvarmm==1);
        marqueurstopmm=find(diffvarmm==-1);
        
        %% premier cycle marche complet ?
        minstartmm=min(marqueurstartmm);
        minstopmm=min(marqueurstopmm);
        if minstartmm<minstopmm
            pccmm=1;
        else
            pccmm=0;
        end
        
        %% dernier cycle marche complet ?
        maxstartmm=max(marqueurstartmm);
        maxstopmm=max(marqueurstopmm);
        if maxstartmm<maxstopmm
            dccmm=1;
        else
            dccmm=0;
        end
        
        %% nombre de cycle marche
        if pccmm==1 && dccmm==1
            nbcyclemm=length(marqueurstartmm);
        elseif pccmm==0 && dccmm==0
            nbcyclemm=length(marqueurstartmm)-2;
        else
            nbcyclemm=length(marqueurstartmm)-1;
        end
        
        %% marqueur cycle marche
        marqueurmm=table();
        for i=1:nbcyclemm
            marqueuraddmm=table();
            if pccmm==1
                marqueuraddmm.cycle=i;
                if varanamm.Time(marqueurstartmm(i,1)+1)-varanamm.Time(marqueurstartmm(i,1))>milliseconds(100)
                    marqueuraddmm.mstart=marqueurstartmm(i,1)+1;
                else
                    marqueuraddmm.mstart=marqueurstartmm(i,1);
                end
                marqueuraddmm.mend=marqueurstopmm(i,1);
            else
                marqueuraddmm.cycle=i;
                if varanamm.Time(marqueurstartmm(i,1)+1)-varanamm.Time(marqueurstartmm(i,1))>milliseconds(100)
                    marqueuraddmm.mstart=marqueurstartmm(i+1,1)+1;
                else
                    marqueuraddmm.mstart=marqueurstartmm(i+1,1);
                end
                marqueuraddmm.mend=marqueurstopmm(i+1,1);
            end %pcc=1
            marqueurmm=[marqueurmm;marqueuraddmm]; %#ok<AGROW>
        end %i=1:nbcycle
        
        %% type de cycle marche
        if ~isempty(marqueurmm)
            cycle_mm.marqueurmm=marqueurmm;
            data_av.ctxt.nb_cycle=height(marqueurmm);
            % Cycle d'ouverture avec fermeture temporisé
            for i=1:height(marqueurmm)
                datacycle=data_av.courbe(marqueurmm.mstart(i,1):marqueurmm.mend(i,1),:);
                ctxtcycle=table;
                if isempty(datacycle)
                    marqueurmm.sys{i,1}='MM';
                    marqueurmm.type{i,1}='E';
                    ctxtcycle.id_type=7;
                    ctxtcycle.typelabel='Cycle vide';
                    marqueurmm.tps(i,1)=0;
                    ctxtcycle.temps_cycle=0;
                else
                    if isnan(nanmean(datacycle.PMM)) || abs(datacycle.PMM(1,1)-datacycle.PMM(end,1))<20
                        marqueurmm.sys{i,1}='MM';
                        marqueurmm.type{i,1}='E';
                        ctxtcycle.id_type=5;
                        ctxtcycle.typelabel='Error deltaP=0';
                    else
                        if nanmax(datacycle.IMM)<100
                            marqueurmm.sys{i,1}='MM';
                            marqueurmm.type{i,1}='E';
                            ctxtcycle.id_type=6;
                            ctxtcycle.typelabel='Error bruit capteur I';
                        else
                            if sum(datacycle.LT_CF)==0 %ouverture
                                if max(datacycle.PMM>200)
                                    marqueurmm.sys{i,1}='MM';
                                    marqueurmm.type{i,1}='O';
                                    ctxtcycle.id_type=1;
                                    ctxtcycle.typelabel='ouverture';
                                else
                                    marqueurmm.sys{i,1}='MM';
                                    marqueurmm.type{i,1}='OC';
                                    ctxtcycle.id_type=2;
                                    ctxtcycle.typelabel='ouverture courte';
                                end
                            else %fermeture
                                if max(datacycle.PMM>180)
                                    marqueurmm.sys{i,1}='MM';
                                    marqueurmm.type{i,1}='F';
                                    ctxtcycle.id_type=3;
                                    ctxtcycle.typelabel='fermeture';
                                else
                                    marqueurmm.sys{i,1}='MM';
                                    marqueurmm.type{i,1}='FC';
                                    ctxtcycle.id_type=4;
                                    ctxtcycle.typelabel='fermeture courte';
                                end
                            end
                        end
                    end
                    marqueurmm.tps(i,1)=duration(datacycle.Time(end)-datacycle.Time(1));
                    ctxtcycle.temps_cycle=seconds(duration(datacycle.Time(end)-datacycle.Time(1)));
                end
                cycle_mm.(['cycle_',num2str(i),'_data'])=datacycle;
                cycle_mm.(['cycle_',num2str(i),'_ctxt'])=ctxtcycle;
                cycle_mm.marqueurmm=marqueurmm;
            end
        else
            data_av.ctxt.nb_cycle=data_av.ctxt.nb_cycle;
            cycle_mm.marqueurmm=table();
        end
        % marqueurtracermm(data_av,ConfigCBM,marqueurmm)
    end
    marqueur=[marqueur;marqueurmm];
catch ME
    error.nb=1;
    error.type='PREPROC';
    error.label=['CATCH : ',ME.message];
    %% erreur inconnue disp console
    %disp(ME.message)
    %disp(struct2table(ME.stack))%             folderfilename=[repfichier,'\',nomfichier];
    %% erreur inconnue copy du fichier
    %movefile(subds.Files{i},[brct_ConfigCBM.repfolder,'CBMCatch'])
    %% erreur inconnue alerte mail
    %setpref('Internet','SMTP_Server','192.168.1.66');
    %setpref('Internet','E_mail','sylvain.grison@sncf.fr');
    %recipients={'ext.ikos.fabien.turgis@sncf.fr'}; %demander a pierre ?
    %strstack='';
    %for k=1:length(ME.stack)
    %    strstackadd=['name : ',ME.stack(k).name,' line : ',num2str(ME.stack(k).line),10,10];
    %    strstack=[strstack,strstackadd]; %#ok<AGROW>
    %end
    %message=['nom du fichier : ',folderfilename,10,10,'ME message :',10,10,ME.message,10,10,'ME stack : ',10,10,strstack];
    %sendmail(recipients,'CBM_IND_AV_NAT : rapport erreur ',message);
end



