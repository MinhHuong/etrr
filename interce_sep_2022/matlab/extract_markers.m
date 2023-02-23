function [error_nb] = extract_markers(out_repo, filename)

% desperate
feature('DefaultCharacterSet', 'UTF8'); %# for all Character support

% gotta decode once again to have the correct format
data = decoder_CBM_File(filename);

% decoupage de cycles & mise au propre
error = struct();
error.nb = 0;
error.label = '';
[data_av, cycle_por, cycle_mm, ~, error] = preprocess(data, filename, error);

% identification des cycles & correction des data
if error.nb == 0
    [~, cycle_por, cycle_mm, error] = cleaning_cycle(data_av, cycle_por, cycle_mm, error);
else
    error_nb = error.nb;
    return
end

% write the results in files
[~, fname, ~] = fileparts(filename);

% create one folder for this file
folder_name = strcat(out_repo, '/', fname);
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

% write markers for porte and marche in separate file
writetable(cycle_mm.marqueurmm, strcat(out_repo, '/', fname, '/', 'marche.csv'));
writetable(cycle_por.marqueur, strcat(out_repo, '/', fname, '/', 'porte.csv'));

% return error number
error_nb = error.nb;

end