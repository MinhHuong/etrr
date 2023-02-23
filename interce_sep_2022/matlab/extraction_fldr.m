function list_cbm_file=extraction_fldr(rep_fldr,rep_bin)
%% fonction permettant de Dziper les fichiers réçus du TCMS communicant

%% Récupération de la liste des tar
lst_fldr = struct2table(dir(rep_fldr{1,1}));
lst_fldr = lst_fldr.name(lst_fldr.isdir==1&contains(lst_fldr.name,'dcupackager','IgnoreCase',true),:);
nb_fldr=length(lst_fldr);
%création list_cbm_file
list_cbm_file=table();

if nb_fldr ==0
    disp('Pas de dossiers à extraire !')
    while nb_fldr==0
        disp('En attente de nouveaux dossiers')
        pause(300)
        lst_fldr = struct2table(dir(rep_fldr{1,1}));
        lst_fldr = lst_fldr.name(lst_fldr.isdir==0 & contains(lst_fldr.name,'dcupackager','IgnoreCase',true),:);
        nb_fldr=length(lst_fldr);
    end
    disp(['Extraction de ',char(string(nb_fldr)),' dossiers'])
end

% Traitement des tar
if nb_fldr>20
    for num_folders=1:20
        
        %nom du tar
        foldername=lst_fldr{num_folders,1};
        lst_file = struct2table(dir([rep_fldr{1,1},'\',foldername]));
        lst_file = lst_file.name(lst_file.isdir==0&contains(lst_file.name,'dcuconddata','IgnoreCase',true),:);
        nb_files=height(lst_file);
        
        for j=1:nb_files
            list_cbm_fileadd=table();
            filename=lst_file{j,1};
            cbm_file=movefile([rep_fldr{1,1},'\',foldername,'\',filename],rep_bin{1,1})';
            if cbm_file==1
                list_cbm_fileadd.cbm_file{1,1}=filename;
                list_cbm_fileadd.cbm_tar{1,1}=foldername;
            else
                %% faire alerte elk fichier erreur de copy
            end
            list_cbm_file=[list_cbm_file;list_cbm_fileadd];  
        end
        rmdir([rep_fldr{1,1},'\',foldername])
    end
else
    for num_folders=1:nb_fldr
        %nom du tar
        foldername=lst_fldr{num_folders,1};
        lst_file = struct2table(dir([rep_fldr{1,1},'\',foldername]));
        lst_file = lst_file.name(lst_file.isdir==0&contains(lst_file.name,'dcuconddata','IgnoreCase',true),:);
        nb_files=height(lst_file);
        
        for j=1:nb_files
            list_cbm_fileadd=table();
            filename=lst_file{j,1};
            cbm_file=movefile([rep_fldr{1,1},'\',foldername,'\',filename],rep_bin{1,1});
            if cbm_file==1
                list_cbm_fileadd.cbm_file{1,1}=filename;
                list_cbm_fileadd.cbm_tar{1,1}=foldername;
            else
                %% faire alerte elk fichier erreur de copy
            end
            list_cbm_file=[list_cbm_file;list_cbm_fileadd];  
        end
        rmdir([rep_fldr{1,1},'\',foldername])
    end
end

disp('Extraction OK !')

end

