%% Read supermag data. The supermag data can be event specific or for the whole year.
%A user defined selected number of stations are downloaded. There are some
%stations missing at certain times. ### We will ignore those stations for the time being.### This is a work in progress code.
clear
ext = '20200529-05-51-supermagDecember2015.txt';
fid0 = fopen(ext);
count = 1;Head = 1;
stnum = -1; hnum = 69; tcnt = 1; data.time = []; data.stnam = []; statnam = []; %header starts at line number 84(69 GR text)
while (isequal(feof(fid0),0))
    data_line = fgetl(fid0);
    %Next block finds header and data values
    if count == hnum + stnum+1
        hnum = count; % header line number
        % NOTE: The data_line length depends on the paramater selected during download. Future version of this code will incorporate those features. 
        % Till then manually change the number where the data and station
        % number is located
%         data.time = vertcat(data.time,data_line(1:19));    % Vertically concatenate time
%         data.time = [data.time,split(data_line(1:19))];    % Concatenate time
%         stnum = str2num(data_line(20:22)); % Number stations selected    
        data.time = [data.time,split(data_line(1:22))];    % Concatenate time
        stnum = str2num(data_line(23:24)); % Number stations selected    
%         data.stnam = vertcat(data.stnam,statnam) ;
        tcnt = tcnt+1;
    end
    if count>hnum && count<hnum+ stnum+1 % Data  from all the stations at a given time
        stn = {data_line(1:3)};
%         statnam = [statnam ;stn];
        if isfield(data,stn{1})== 0
%             data.(stn{1}) = str2double(data_line(4:end));
            data.(stn{1}) = split(data_line(5:end));  % Extract space delimited data
        else
%             data.(stn{1}) = vertcat(data.(stn{1}),str2double(data_line(4:end)));
%             data.(stn{1}) = vertcat(data.(stn{1}),split(data_line(5:end)));
            data.(stn{1}) = [data.(stn{1}),split(data_line(5:end))];
        end 
    end
    
    count = count +1;

end
fclose(fid0);

save