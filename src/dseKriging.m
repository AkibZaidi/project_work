%rotation 90 degrees
basePath = "C:/TUD_Course/ADS_Project_Work/Academic_Project/pw_zaidi/Sourcecode" + ...
"";
fileName = "";

% The complete search pathe is generated from the base path and the
% fileName.
tableSearchPath = basePath + fileName;
dbPath = basePath + '/db/mat/'

% an excel file can contain mutliple tables. To each tbale, a name is
% assigned in the original file. All tables of the test file are listed 
% with their respective name and data range in the following lines. This
% was made to speed up testing. One might want to adapt the following lines
% for other files.
pktLength = transpose(load(dbPath+'Bitcomplement.mat','packetLength').packetLength);
buffSizePy = transpose(load(dbPath+'Bitcomplement.mat','bufferSize').bufferSize);
timePerFlitPy = transpose(load(dbPath+'Bitcomplement.mat','timePerFlit').timePerFlit);
plot_title = "Bitcomplement"
clear variables
clc
observations.x=[1 0.5 2.5 4 3 2];
observations.y=[1 2 4 3 1 2];
observations.z=[0 0 0 0 0 1];
options.param='off';
figure('Name','12.6')
krigeage(observations,options);
%view(2)
