function [error_nb] = preprocess_cycle(filename, out_repo)

% desperate
feature('DefaultCharacterSet', 'UTF8'); % for all Character support

% decode the binary content
data_av = decoder_CBM_File(filename);

% check for error before writing
if data_av.error.nb ~= 0
    error = data_av.error;
    error_nb = error.nb;
    return
end

% write the results in files
[~, fname, ~] = fileparts(filename);

% create one folder for this filename
mkdir(strcat(out_repo, '/', fname));

% create a file for the global context
ctxt_fname = strcat(out_repo, '/', fname, '/', 'ctxt.csv');
writetable(data_av.ctxt, ctxt_fname);

% save the decoded data in CSV
decod_fname = strcat(out_repo, '/', fname, '/', 'decoded.csv');
writetimetable(data_av.courbe, decod_fname);

% return the error number
error = data_av.error;
error_nb = error.nb;


end