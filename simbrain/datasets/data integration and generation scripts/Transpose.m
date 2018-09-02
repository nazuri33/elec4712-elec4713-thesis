output = [];
countx = 1;
county = 1;

for i = 1:length(VarName4)
    if (VarName4(i) == 100)
        countx = countx + 1;
        county = 1;
    else
        output(countx,county) = VarName4(i);
        county = county + 1;
    end
end