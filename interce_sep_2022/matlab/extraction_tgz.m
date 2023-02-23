function list_cbm_file=extraction_tgz(rep_tar,rep_bin)
%% fonction permettant de Dziper les fichiers réçus du TCMS communicant

%% Récupération de la liste des tar
lst_files = struct2table(dir(rep_tar{1,1}));
lst_files = lst_files.name(lst_files.isdir==0&contains(lst_files.name,'.tgz','IgnoreCase',true),:);
nb_files=length(lst_files);
%création list_cbm_file
list_cbm_file=table();

if nb_files ==0
    disp('Pas de fichiers .tgz à extraire !')
    while nb_files==0
        disp('En attente de nouveaux fichiers')
        pause(300)
        lst_files = struct2table(dir(rep_tar{1,1}));
        lst_files = lst_files.name(lst_files.isdir==0 & contains(lst_files.name,'.tgz','IgnoreCase',true),:);
        nb_files=length(lst_files);
    end
    disp(['Extraction de ',char(string(nb_files)),' fichiers .tar'])
end

% Traitement des tar
if nb_files>20
    for num_files=1:20
        list_cbm_fileadd=table();
        %nom du tar
        tarname=char(string(lst_files(num_files)));
        try
            cbm_file=untar([rep_tar{1,1},'\',tarname],rep_bin{1,1})';
            for j=1:length(cbm_file)
                splitcbmfile=strsplit(cbm_file{j,1},'\');
                list_cbm_fileadd.cbm_file(j,1)=splitcbmfile(end);
                list_cbm_fileadd.cbm_tar{j,1}=tarname;
            end
            list_cbm_file=[list_cbm_file;list_cbm_fileadd];
        catch me
            delete([rep_tar{1,1},'\',tarname])
            continue %passe au num_files suivant
        end
    end
else
    for num_files=1:nb_files
        list_cbm_fileadd=table();
        %nom du tar
        tarname=char(string(lst_files(num_files)));
        try
            cbm_file=untar([rep_tar{1,1},'\',tarname],rep_bin{1,1})';
            for j=1:length(cbm_file)
                splitcbmfile=strsplit(cbm_file{j,1},'\');
                list_cbm_fileadd.cbm_file(j,1)=splitcbmfile(end);
                list_cbm_fileadd.cbm_tar{j,1}=tarname;
            end
            list_cbm_file=[list_cbm_file;list_cbm_fileadd];
        catch me
            delete([rep_tar{1,1},'\',tarname])
            continue %passe au num_files suivant
        end
    end
end

disp('Extraction OK !')

end

