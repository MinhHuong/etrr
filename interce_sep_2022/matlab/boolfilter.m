function dataout=boolfilter(datain)
nbpt=length(datain);
for i=2:nbpt-1
    if datain(i-1)==1 && datain(i)==0 && datain(i+1)==1
        datain(i)=1;
    end
    if datain(i-1)==0 && datain(i)==1 && datain(i+1)==0
        datain(i)=0;
    end
end %i=1:nbpt
for i=2:nbpt-1
    if datain(i-1)==1 && datain(i)==0 && datain(i+1)==1
        datain(i)=1;
    end
    if datain(i-1)==0 && datain(i)==1 && datain(i+1)==0
        datain(i)=0;
    end
end %i=1:nbpt
dataout=datain;
end % function