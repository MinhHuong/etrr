function crc_test = crc_ccitt_matlab (data)
crc_test=0;
message=data(1:end-2);
    %crc = uint16(hex2dec('0000'));
    crc = uint16(0);

    for i = 1:length(message)
        crc = bitxor(crc,bitshift(message(i),8));

        for j = 1:8
           % if (bitand(crc, hex2dec('8000')) > 0)
            if (bitand(crc, 32768) > 0)
                 crc = bitxor(bitshift(crc, 1), 4129);
               % crc = bitxor(bitshift(crc, 1), hex2dec('1021'));
            else
                crc = bitshift(crc, 1);
            end
        end
    end

    crc_val = crc;
    a=typecast(crc_val,'uint8');
    if a(1)==data(end-1) && a(2)==data(end)
        crc_test=1;
    end
end