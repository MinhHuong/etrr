function [data_av,cycle_por,cycle_mm,error]=cleaning_cycle(data_av,cycle_por,cycle_mm,error)

try
    if ~isempty(cycle_por.marqueur)
        for i=1:height(cycle_por.marqueur)
            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=0;
            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='';
            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='no error';
            data=cycle_por.(['cycle_',num2str(i),'_data']);
            ctxt=cycle_por.(['cycle_',num2str(i),'_ctxt']);
            
            %% check continues acquisition
            deltaqu=milliseconds(diff(data.Time));
            if sum(deltaqu>50)>0
                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=2;
                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='acquisition non continue';
            end
            
            switch ctxt.id_type
                case 1 % ouverture
                    %% check position offset
                    if isnan(data.PPOR(1,1))
                        nanval=find(isnan(data.PPOR));
                        replaceval=data.PPOR(max(nanval)+1,1);
                        data.PPOR(nanval,1)=replaceval;
                    end
                    pstart=data.PPOR(1,1);
                    pneg=find(data.PPOR<-7, 1);
                    if pstart>150 && pstart<250
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=3;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='grany test';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'3']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','grany test'];
                        end
                    else
                        if pstart>7 || pstart<-7
                            if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=4;
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='offset position';
                            else
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'4']);
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','offset position'];
                            end
                        end
                        %% ajustement acquisition
                        % remplissage position début de cycle
                        if isnan(data.IPOR(1,1))
                            nanval=find(isnan(data.IPOR));
                            replaceval=data.IPOR(max(nanval)+1,1);
                            data.IPOR(nanval,1)=replaceval;
                        end
                        if isnan(data.TPOR(1,1))
                            nanval=find(isnan(data.TPOR));
                            replaceval=data.TPOR(max(nanval)+1,1);
                            data.TPOR(nanval,1)=replaceval;
                        end
                        % enregistrement point de référence
                        if data.IPOR(1,1)~=0 || data.TPOR(1,1)~=0
                            datainit=data(1,:);
                            datainit.Time=datainit.Time-milliseconds(50);
                            datainit.IPOR=0;
                            datainit.TPOR=0;
                            data=[datainit;data];
                        end
                        if data.IPOR(end,1)~=0 || data.TPOR(end,1)~=0
                            datainit=data(end,:);
                            datainit.Time=datainit.Time+milliseconds(50);
                            datainit.IPOR=0;
                            datainit.TPOR=0;
                            data=[data;datainit];
                        end
                        cycle_por.(['cycle_',num2str(i),'_data'])=data;
                        %% temps de cycle trop court
                        if ctxt.temps_cycle<2.5
                            if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=5;
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle court';
                            else
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'5']);
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                            end
                        end
                        %% temps de cycle trop long
                        % supp controle ltao ?==1
                        ctxt.temps_cycle=seconds(data.Time(end)-data.Time(1)); %catch z5700019________z_57_00019______dcu2____________dcuconddata______201117_154632-error=4_PREPROC_cycle vide
                        if ctxt.temps_cycle>5
                            if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=6;
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle long';
                            else
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'6']);
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                            end
                        else
                            cycle_por.(['cycle_',num2str(i),'_data'])=data;
                            cycle_por.(['cycle_',num2str(i),'_ctxt'])=ctxt;
                        end
                        %% check position negative
                        if ~isempty(pneg)
                            if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=7;
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='negative position';
                            else
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'7']);
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','negative position'];
                            end
                        end
                        %% check cycle incomplet
                        pdelta=abs(data.PPOR(1)-data.PPOR(end));
                        if pdelta<750
                            if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=8;
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='incomplete position';
                            else
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'8']);
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                                cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','incomplete position'];
                            end
                        end
                    end
                case 2 % fermeture
                    % enregistrement point de référence
                    if data.IPOR(1,1)~=0 || data.TPOR(1,1)~=0
                        datainit=data(1,:);
                        datainit.Time=datainit.Time-milliseconds(50);
                        datainit.IPOR=0;
                        datainit.TPOR=0;
                        data=[datainit;data];
                    end
                    if data.IPOR(end,1)~=0 || data.TPOR(end,1)~=0
                        datainit=data(end,:);
                        datainit.Time=datainit.Time+milliseconds(50);
                        datainit.IPOR=0;
                        datainit.TPOR=0;
                        data=[data;datainit];
                    end
                    cycle_por.(['cycle_',num2str(i),'_data'])=data;
                    %% temps de cycle trop court
                    if ctxt.temps_cycle<2.5
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=3;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle court';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'3']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                        end
                    end
                    %% temps de cycle trop long
                    if ctxt.temps_cycle>5
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=4;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle long';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'4']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                        end
                    end
                    %% check position negative
                    pneg=find(data.PPOR<-7, 1);
                    if ~isempty(pneg)
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=5;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='negative position';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'5']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','negative position'];
                        end
                    end
                    %% check cycle incomplet
                    pdelta=abs(data.PPOR(1)-data.PPOR(end));
                    if pdelta<750
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=6;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='incomplete position';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'6']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','incomplete position'];
                        end
                    end
                case 3 %fermeture lente
                    % enregistrement point de référence
                    if data.IPOR(1,1)~=0 || data.TPOR(1,1)~=0
                        datainit=data(1,:);
                        datainit.Time=datainit.Time-milliseconds(50);
                        datainit.IPOR=0;
                        datainit.TPOR=0;
                        data=[datainit;data];
                    end
                    if data.IPOR(end,1)~=0 || data.TPOR(end,1)~=0
                        datainit=data(end,:);
                        datainit.Time=datainit.Time+milliseconds(50);
                        datainit.IPOR=0;
                        datainit.TPOR=0;
                        data=[data;datainit];
                    end
                    cycle_por.(['cycle_',num2str(i),'_data'])=data;
                    %% temps de cycle trop court
                    if ctxt.temps_cycle<7
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=3;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle court';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'3']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                        end
                    end
                    %% temps de cycle trop long
                    if ctxt.temps_cycle>9
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=4;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle long';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'4']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle court'];
                        end
                    end
                    %% check cycle incomplet
                    pdelta=abs(data.PPOR(1)-data.PPOR(end));
                    if pdelta<750
                        if cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=5;
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='incomplete position';
                        else
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle),'5']);
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                            cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','incomplete position'];
                        end
                    end
                otherwise
                    cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle=9;
                    cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                    cycle_por.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='error identification cycle';
            end %switch ctxt.type
            
        end %for i=1:height(cycle_por.marqueur)
    end %if ~isempty(cycle_por.marqueur)
    %% marche mobile
    if ~isempty(cycle_mm.marqueurmm)
        for i=1:height(cycle_mm.marqueurmm)
            cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=0;
            cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='';
            cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='no error';
            data=cycle_mm.(['cycle_',num2str(i),'_data']);
            ctxt=cycle_mm.(['cycle_',num2str(i),'_ctxt']);
            
            %% check continues acquisition
            deltaqu=diff(data.Time);
            if sum(deltaqu>50)>0
                if cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=2;
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='acquisition non continue';
                else
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle),'2']);
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                    cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','acquisition non continue'];
                end
            end
            
            if ctxt.id_type==1 || ctxt.id_type==2 % ouverture
                if isnan(data.PMM(1,1))
                    nanval=find(isnan(data.PMM));
                    replaceval=data.PMM(max(nanval)+1,1);
                    data.PMM(nanval,1)=replaceval;
                end
                if isnan(data.IMM(1,1))
                    nanval=find(isnan(data.IMM));
                    replaceval=data.IMM(max(nanval)+1,1);
                    data.IMM(nanval,1)=replaceval;
                end
                if isnan(data.TMM(1,1))
                    nanval=find(isnan(data.TMM));
                    replaceval=data.TMM(max(nanval)+1,1);
                    data.TMM(nanval,1)=replaceval;
                end
                % enregistrement point de référence
                if data.IMM(1,1)~=0 || data.TMM(1,1)~=0
                    datainit=data(1,:);
                    datainit.Time=datainit.Time-milliseconds(50);
                    datainit.IMM=0;
                    datainit.TMM=0;
                    data=[datainit;data];
                end
                if data.IMM(end,1)~=0 || data.TMM(end,1)~=0
                    datainit=data(end,:);
                    datainit.Time=datainit.Time+milliseconds(50);
                    datainit.IMM=0;
                    datainit.TMM=0;
                    data=[data;datainit];
                end
                cycle_mm.(['cycle_',num2str(i),'_data'])=data;
                %% temps de cycle trop long
                if ctxt.temps_cycle>4
                    if cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=3;
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle long';
                    else
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle),'3']);
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle long'];
                    end
                end
                %% check position negative
                pneg=find(data.PMM<-7, 1);
                if ~isempty(pneg)
                    if cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=4;
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='negative position';
                    else
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle),'4']);
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','negative position'];
                    end
                end
            else % fermeture
                % enregistrement point de référence
                if data.IMM(1,1)~=0 || data.TMM(1,1)~=0
                    datainit=data(1,:);
                    datainit.Time=datainit.Time-milliseconds(50);
                    datainit.IMM=0;
                    datainit.TMM=0;
                    data=[datainit;data];
                end
                if data.IMM(end,1)~=0 || data.TMM(end,1)~=0
                    datainit=data(end,:);
                    datainit.Time=datainit.Time+milliseconds(50);
                    datainit.IMM=0;
                    datainit.TMM=0;
                    data=[data;datainit];
                end
                cycle_mm.(['cycle_',num2str(i),'_data'])=data;
                %% temps de cycle trop long
                if ctxt.temps_cycle>5
                    if cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=5;
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='tps cycle long';
                    else
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle),'5']);
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','tps cycle long'];
                    end
                end
                %% check position negative
                pneg=find(data.PMM<-7, 1);
                if ~isempty(pneg)
                    if cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle==0
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=6;
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}='negative position';
                    else
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle=str2double([num2str(cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle),'6']);
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_type{1,1}='MARK MM';
                        cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1}=[cycle_mm.(['cycle_',num2str(i),'_ctxt']).error_cycle_label{1,1},',','negative position'];
                    end
                end
            end %if ctxt.type==1
        end %for i=1:height(cycle_mm.marqueurmm)
    end %if ~isempty(cycle_mm.marqueurmm)
    
catch ME
    error.nb=1;
    error.type='MARK';
    error.label=['CATCH : ',ME.message];
    %% erreur inconnue disp console
    disp(ME.message)
    disp(struct2table(ME.stack))%             folderfilename=[repfichier,'\',nomfichier];
    %% erreur inconnue alerte mail
    setpref('Internet','SMTP_Server','192.168.1.66');
    setpref('Internet','E_mail','sylvain.grison@sncf.fr');
    recipients={'ext.ikos.fabien.turgis@sncf.fr'}; %demander a pierre ?
    strstack='';
    for k=1:length(ME.stack)
        strstackadd=['name : ',ME.stack(k).name,' line : ',num2str(ME.stack(k).line),10,10];
        strstack=[strstack,strstackadd]; %#ok<AGROW>
    end
    message=['nom du fichier : ','folderfilename',10,10,'ME message :',10,10,ME.message,10,10,'ME stack : ',10,10,strstack];
    sendmail(recipients,'CBM_IND_AV_NAT : rapport erreur ',message);
end