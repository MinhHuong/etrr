function [data_av] = decoder_CBM_File(filename)

if nargin==0
    
    %  nomfichier='CBM_Logical1.bin';
    % filename='CBM_Logical2.bin';
    %     nomfichier='CBM_Logical3.bin';
    %     nomfichier='CBM_Logical4.bin';
    %     nomfichier='CBM_Open-Close_Analog-Data.bin';
    
end
try
    data_av=struct();
    data_av.error=table();
    data_av.error.nb=0;
    data_av.error.type{1,1}='';
    data_av.error.label{1,1}='';
    ctxt=table();
    load ('referentiel_CD.mat','referentiel_CD');
    
    fid=fopen(filename,'r');
    
    if fid==-1
        %    cd(rep_exe);
        disp('probleme de lecture du fichier brut')
        data_av.error.nb=2;
        data_av.error.type='DECOD';
        data_av.error.label='probleme de lecture du fichier brut';
        return
    end
    
    data = fread(fid); % PPermet de convertir directement de l'ASCII etendu en decimal
    fclose(fid);
    %%
    % test du crc a coder
    if ~isempty(data)
        if ~crc_ccitt_matlab(data)
            data_output=timetable(); %#ok<NASGU>
            data_av.error.nb=2;
            data_av.error.type='DECOD';
            data_av.error.label='fichiercoromtpu';
            disp('fichiercoromtpu');
            return
        end

        % decodage du header
        %%List of data saved in standard header;%%
        Major_version=data(1,1);
        Minor_version=data(2,1);
        Number_of_bytes_allocated_for_IO_snapshot=data(3,1);
        Number_of_bytes_allocated_for_Analog_snapshot=data(4,1);
        Number_of_snapshot_variable=data(6,1)*256+data(5,1);
        Number_of_specific_variable=data(8,1)*256+data(7,1);
        Size_of_cyclique_record_area_in_bytes=data(12)*256^3 + data(11)*256^2 +  data(10)*256 + data(9)*1;
        Number_of_record=data(16)*256^3 + data(15)*256^2 + data(14)*256 + data(13)*1;
        CBM_startup_time=data(24)*256^7 + data(23)*256^6 + data(22)*256^5 + data(21)*256^4 +data(20)*256^3 + data(19)*256^2 + data(18)*256 + data(17)*1;
        Snapshot_period=data(26)*256 + data(25)*1;
        
        %%
        % decodage des parametres des variables loggees
        cpt_end=4161;
        cpt_start=65;
        cpt=cpt_start;
        name_var=cell(Number_of_snapshot_variable,1);
        size_var=zeros(Number_of_snapshot_variable,1);
        octet_var_idx=zeros(Number_of_snapshot_variable,1);
        bit_var_idx=zeros(Number_of_snapshot_variable,1);
        
        for  nb_bloc= 1 : Number_of_snapshot_variable
            taille_bloc=data(cpt);
            name_var{nb_bloc}={char(data(cpt+1:cpt+taille_bloc-4-2))};
            octet_var_idx(nb_bloc,1)=data(cpt+taille_bloc-2);
            bit_var_idx(nb_bloc,1)=data(cpt+taille_bloc-3);
            size_var(nb_bloc,1)=data(cpt+taille_bloc-4);
            cpt=cpt+taille_bloc;
        end
        
        %%
        % decodage du cotexte
        %
        % 1 = BOOLEAN(V=1), 2 = UINT08 (V=1),3 = SINT08 (V=1), 4 = UINT16 (V=2),
        % 5 = SINT16 (V=2), 6= UINT32 (V=4), 7 = SINT32 (V=4), 8= UINT64 (V=8), 9 = SINT64
        % (V=8), 10 = FLOAT32 (V=4), 11 = FLOAT64 (V=8), 12 = STRING 
        ctx_name_var=cell(Number_of_specific_variable,1);
        ctx_size_var=zeros(Number_of_specific_variable,1);
        %ctx_data_var=zeros(Number_of_specific_variable,1);
        ctx_data_cell=cell(Number_of_specific_variable,1);
        
        for  nb_bloc= 1 : Number_of_specific_variable   
            taille_bloc=data(cpt);
            %data_bloc= data(cpt+1:cpt+taille_bloc);
            idx_end_name=(find(data(cpt+1:cpt+taille_bloc)==0,1));
            
            ctx_name_var{nb_bloc,1}={char(data(cpt+1:cpt+taille_bloc-4-2))};
            % taille_data=
            type_var=data(idx_end_name+cpt+1);
            %ctx_type_var(nb_bloc,1)=type_var;
            ctx_size_var(nb_bloc,1)=data(idx_end_name+cpt+2);
            ctx_size_var_=data(idx_end_name+cpt+2);
            switch type_var
                case {1, 2,3}
                    Madata = data(cpt+taille_bloc-2);
                case 4
                    Madata =typecast(uint8([data(cpt+taille_bloc-3),data(cpt+taille_bloc-2)]), 'UINT16');
                case 5
                    Madata =typecast(uint8([data(cpt+taille_bloc-3),data(cpt+taille_bloc-2)]), 'INT16');
                case {6,7}
                    Madata =typecast(uint8([data(cpt+idx_end_name+3),data(cpt+idx_end_name+4),data(cpt+idx_end_name+5),data(cpt+idx_end_name+6)]), 'UINT32');
                case {8,9,11}
                    Madata =typecast(uint8([data(cpt+idx_end_name+3),data(cpt+idx_end_name+4),data(cpt+idx_end_name+5),data(cpt+idx_end_name+6),data(cpt+idx_end_name+7),data(cpt+idx_end_name+8),data(cpt+idx_end_name+9),data(cpt+idx_end_name+10)]), 'UINT64');
                    
                case 10
                    Madata =typecast(uint8([data(cpt+idx_end_name+3),data(cpt+idx_end_name+4),data(cpt+idx_end_name+5),data(cpt+idx_end_name+6)]), 'single');
                case {12}
                    Madata = num2str(data(cpt+idx_end_name+3:cpt+idx_end_name+2+ctx_size_var_));
                otherwise
                    disp('type de donnee inconnu');
                    Madata =NaN;
            end
            
            ctx_data_cell{nb_bloc,1}=Madata;
            cpt=cpt+taille_bloc;
            
        end
        
        ctx_data_cell = context_fix_r2n (ctx_data_cell);
        
        %%
        % decodage des donnees cyclique
        
        cpt = cpt_end;
        timestamp=zeros(Number_of_record,1);
        Transition=zeros(Number_of_record,1);
        SleepMode=zeros(Number_of_record,1);
        ID_=zeros(Number_of_record,1);
        
        %data_cbm=zeros(Number_of_record,64);
        data_cbm=nan(Number_of_record,64);
        
        for i=1 :Number_of_record
            ID=data(cpt);
            AA=dec2hex(ID);
            ID_(i,1)= ID;
            Transition(i,1)= str2double(AA(1));
            SleepMode(i,1)=str2double(AA(2));
            timestamp(i,1)=data(cpt+1)+data(cpt+2)*256+data(cpt+3)*65536;
            dim_zipcontrol=sum(de2bi(data(cpt+4)));
            zipcontrol=de2bi(data(cpt+4),8);
            
            data_block_mask=data(cpt+5:cpt+4+dim_zipcontrol);
            
            cpt_octet_change=0;
            cpt_value_change=0;
            for bit_zip_control=1:8
                if zipcontrol(bit_zip_control)==1
                    cpt_octet_change=cpt_octet_change+1;
                    
                    %octet a change
                    masck_octet=data_block_mask(cpt_octet_change);
                    bin_mask_octet=de2bi(masck_octet,8);
                    for bbit_zip_cintrol =1:8
                        nb_colonne=bit_zip_control*8-8+bbit_zip_cintrol;
                        if bin_mask_octet(bbit_zip_cintrol)==1
                            %bit a change
                            cpt_value_change=cpt_value_change+1;
                            data_value=data(cpt+4+dim_zipcontrol+cpt_value_change);
                            
                            data_cbm(i,nb_colonne)=data_value;
                            
                        else
                            %bit n'a pas change
                            if i==1
                                data_cbm(i,nb_colonne)=0;
                            else
                                data_cbm(i,nb_colonne)=data_cbm(i-1,nb_colonne);
                            end
                        end
                    end
                    
                else
                    %octect n'a pas change
                    if i==1
                        data_cbm(i,bit_zip_control*8-7:bit_zip_control*8)=NaN;%old value
                    else
                        data_cbm(i,bit_zip_control*8-7:bit_zip_control*8)=data_cbm(i-1,bit_zip_control*8-7:bit_zip_control*8);
                    end
                end
                
            end
            cpt=cpt+dim_zipcontrol+1+4+cpt_value_change;
        end
        
        %%
        % Mise en forme des donnees avec le nom des colonnes
        
        %Data2_cbm=zeros(Number_of_record,Number_of_snapshot_variable);
        Data2_cbm=nan(Number_of_record,Number_of_snapshot_variable);
        for  nb_var= 1 : Number_of_snapshot_variable
            octet_v= octet_var_idx(nb_var);
            bit_v=bit_var_idx(nb_var);
            %name_v=name_var{nb_var};
            size_v=size_var(nb_var);
            switch size_v
                case 0
                    %boolean
                    boolean_v=de2bi(data_cbm(:,octet_v+1),8);
                    Data2_cbm(:,nb_var)= boolean_v(:,bit_v+1);
                case 4
                    data_cbm(isnan(data_cbm(:,octet_v+2)),octet_v+2)=0;
                    Data2_cbm(:,nb_var)=data_cbm(:,octet_v+1)+data_cbm(:,octet_v+2)*256;
                case 2 %cas rerng
                    data_cbm(isnan(data_cbm(:,octet_v+2)),octet_v+2)=0;
                    Data2_cbm(:,nb_var)=data_cbm(:,octet_v+1)+data_cbm(:,octet_v+2)*256;
                case 5
                    %signe
                    data_cbm(isnan(data_cbm(:,octet_v+2)),octet_v+2)=0;
                    a=data_cbm(:,octet_v+1)+data_cbm(:,octet_v+2)*256;
                    a(a>=32768)=a(a>=32768)-65536;
                    Data2_cbm(:,nb_var)=a;
                case 6 %cas rerng
                    %signe
                    Data2_cbm(:,nb_var)=sum(data_cbm(:,octet_v+1:octet_v+8),2);
                case 8
                    % cas des code defau
                    A=data_cbm(:,octet_v+1:octet_v+8);
                    A(isnan(A)) = 0;
                    for jj=1:size(data_cbm,1)
                        Data2_cbm(jj,nb_var)=typecast(uint8(A(jj,:)),'UINT64');
                    end
                    % pour identifier le bon code" find(de2bi(536870912,64))
                    
                otherwise
                    
                    disp('type de donnee inconnu');
            end
            
        end
        
        t=duration(zeros(Number_of_record,1),zeros(Number_of_record,1),zeros(Number_of_record,1),timestamp);
        table_type=cell(Number_of_snapshot_variable,1);
        table_type(:,1)={'double'};
        %mise en forme de nom
        newname=cell(Number_of_snapshot_variable,1);
        for uu=1 : Number_of_snapshot_variable
            newname{uu,1}=replace(name_var{uu,1}{:}',{'-',' ','é','<',char(34),char(39),'ç',':','.'},'_');
            
        end
        
        data_output=timetable('Size',[Number_of_record Number_of_snapshot_variable],'VariableNames',newname ,'VariableTypes', table_type','rowTimes',t);
        data_output{:,:}=Data2_cbm;
        data_output.SleepMode=SleepMode;
        data_output.Transition=Transition;
        % if sum(Data_output.Position_moteur_de_la_porte<0)>1
        %     erreur_codeur=1;
        % end
        
        % init de variable a 0
        [B(:,1),B(:,2),~] = find(ismissing(data_output));
        [~,ia,~] = unique(B(:,2),'last');
        fist_nan_by_zero=B(ia,:);
        fist_nan_by_zero(fist_nan_by_zero(:,1)==size(data_output,1),:)=[];%suppresion descolonne entirement nan
        for jj=1:size(fist_nan_by_zero,1)
            data_output(fist_nan_by_zero(jj,1),fist_nan_by_zero(jj,2))={0};
        end
        
        CD=de2bi(data_output.Code_du_d_faut_pr_sent__Cf_Tableau_des_defauts,64);
        [~,col]=find(CD);
        col=unique(col);
        str_cd="";
        for jj=1:size(col,1)
            str_cd=strcat(str_cd,'_',referentiel_CD(col(jj),2));
        end
        
        %header
        ctxt.Major_version=Major_version;
        ctxt.Minor_version=Minor_version;
        ctxt.Nb_IO=Number_of_bytes_allocated_for_IO_snapshot;
        ctxt.Nb_Analog=Number_of_bytes_allocated_for_Analog_snapshot;
        ctxt.Nb_variable=Number_of_snapshot_variable;
        ctxt.Nb_ctxt=Number_of_specific_variable;
        ctxt.Size_bytes=Size_of_cyclique_record_area_in_bytes;
        ctxt.Nb_record=Number_of_record;
        ctxt.start_time=CBM_startup_time;
        ctxt.period=Snapshot_period;
        %contexte
        ctxt.serial_number=ctx_data_cell(1);
        ctxt.Device_car=ctx_data_cell{2};
        ctxt.Device_type=ctx_data_cell{3};
        try
            code=ctx_data_cell{4};
            ctxt.Customer_Code=strcat(char(str2double(strcat(code(1),code(18)))),...
                char(str2double(strcat(code(2),code(19)))),char(str2double(strcat(code(3),code(20)))),char(str2double(strcat(code(4),code(21)))));
        catch
            ctxt.Customer_Code='null';
        end
        ctxt.Previous_station_id{1,1}=rot90(deblank(ctx_data_cell{5}),2);
        ctxt.Next_station_id{1,1}=rot90(deblank(ctx_data_cell{6}),2);
        ctxt.Outside_Temp=ctx_data_cell{7};
        ctxt.CD=str_cd;
        data_av.ctxt=ctxt;
        data_output.Properties.VariableNames={'LT_V_DVR','LT_V_2','LT_AO1','LT_AO2','LT_Ads','LT_Acq','LT_CF','LT_CF2','LT_IH_UFR','LT_IF','LT_W','LT_OC','FDCV','FDCF','FDCC1','FDCC2','FDC_CLA_CpF','FDC_CLA_F','FDC_CLA_I','FDC_CLA_C1','FDC_CLA_C2','BPO_I','BPO_E','BP_DU','Carre_CLA','Carre_F_CLA','Poigne_DU_P','Carre_DU_POR_I','Carre_DU_POR_E','Vo_POR_NO_I','Vo_POR_NO_E','Vo_BPO_I','Vo_BPO_E','Vo_BP_DU','Vo_POR_NO_I_Cli','Vo_POR_NO_E_Cli','Vo_UFR_I_Cli','Vo_UFR_E_Cli','Vo_F_Cli','Vo_BPO_I_Cli','Vo_BPO_E_Cli','Vo_BP_DU_Cli','P_defaut','Code_defaut','IPOR','TPOR','PPOR','IMM','TMM','PMM','SleepMode','Transition'};
        data_av.courbe=data_output;
        
        %invest mode sleep
        %     FM_sleep=find(data_output.SleepMode(2:end)-data_output.SleepMode(1:end-1)==1);
        %     FD_sleep=find(data_output.SleepMode(2:end)-data_output.SleepMode(1:end-1)==-1);
        %
        %     if min(data_output.PPOR)<-4 && min(data_output.PPOR)>-255
        %      %   disp([ 'fichier : ' filename(33:end)]);
        %                 disp(['fichier : ' filename(33:end) ' pb codeur porte ,position mini =  '  num2str(min(data_output.PPOR)) ]);
        %     end
        %
        %     if min(data_output.PMM)<-4
        %     %   disp([ 'fichier : ' filename(33:end)]);
        %                 disp(['fichier : ' filename(33:end) ' pb codeur marche ,position mini =  '  num2str(min(data_output.PMM)) ]);
        %     end
        %
        %     for jj=1:length(FD_sleep)
        %         if (~isnan(data_output.PMM(FD_sleep(jj)+1)) && ~isnan(data_output.PMM(FD_sleep(jj))))
        %             nok_M= data_output.PMM(FD_sleep(jj))- data_output.PMM(FD_sleep(jj)+1)>1;
        %             if nok_M
        %                % disp([ 'fichier : ' filename(33:end)]);
        %                 disp(['fichier : ' filename(33:end) ' pb en sortie du mode sleep analog perte de donnée Marche enre la position '  num2str(data_output.PMM(FD_sleep(jj))) ' et la position '   num2str(data_output.PMM(FD_sleep(jj)+1))]);
        %             end
        %         end
        % %         if (~isnan(data_output.Position_moteur_de_la_porte(FD_sleep(jj)+1)) && ~isnan(data_output.Position_moteur_de_la_porte(FD_sleep(jj))))
        % %             nok_P= data_output.Position_moteur_de_la_porte(FD_sleep(jj))- data_output.Position_moteur_de_la_porte(FD_sleep(jj)+1)>1;
        % %             if nok_P
        % %                 disp([ 'fichier : ' filename(33:end)]);
        % %                 disp(['fichier : ' filename(33:end) ' pb en sortie du mode sleep analog perte de donnée Porte enre la position '  num2str(data_output.Position_moteur_de_la_porte(FD_sleep(jj))) ' et la position '   num2str(data_output.Position_moteur_de_la_porte(FD_sleep(jj)+1))]);
        % %             end
        % %         end
        %     end
        %     for jj=1:length(FM_sleep)
        %
        %         if (~isnan(data_output.PMM(FM_sleep(jj)+1)) && ~isnan(data_output.PMM(FM_sleep(jj))))
        %             nok_M= data_output.PMM(FM_sleep(jj))- data_output.PMM(FM_sleep(jj)+1)>1;
        %             if nok_M
        %                 disp([ 'fichier : ' filename(33:end)]);
        %                 disp(['pb en entree du mode sleep analog perte de doneée Marche enre la position '  num2str(data_output.PMM(FM_sleep(jj))) ' et la position '   num2str(data_output.PMM(FM_sleep(jj)+1))]);
        %             end
        %         end
        %         if  (~isnan(data_output.PPOR(FM_sleep(jj)+1)) && ~isnan(data_output.PPOR(FM_sleep(jj))))
        %             nok_P= data_output.PPOR(FM_sleep(jj))- data_output.PPOR(FM_sleep(jj)+1)>1;
        %             if nok_P
        %                 disp([ 'fichier : ' filename(33:end)]);
        %                 disp(['pb en entree du mode sleep analog perte de donnée Porte enre la position '  num2str(data_output.PPOR(FM_sleep(jj))) ' et la position '   num2str(data_output.PPOR(FM_sleep(jj)+1))]);
        %             end
        %         end
        %
        %     end
        
        
    end
catch ME
    data_av.error.nb=1;
    data_av.error.type='DECOD';
    data_av.error.label=['CATCH : ', ME.message];
end
end
function nexwctxt = context_fix_r2n (ctxt)
nexwctxt=ctxt;
weights2 = 2.^([7:-1:0]); %#ok<NBRAK>
num=double(ctxt{5});
bits = reshape(bitget(num,32:-1:1),8,[]); %// num is the input number
out = weights2*bits;
nexwctxt{5}=char(rot90(out,2));

num=double(ctxt{6});
bits = reshape(bitget(num,32:-1:1),8,[]); %// num is the input number
out = weights2*bits;
nexwctxt{6}=char(rot90(out,2));
code = ctxt{4};
end