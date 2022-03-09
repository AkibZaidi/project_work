%% Set up the Import Options and import the data
	opts = spreadsheetImportOptions("NumVariables", 7);
	
	%  ------------                Excel sheet setup         -----------------
	% We use an excel sheet as data input. Configuration to use the right file
	% and sheet can be done in the following lines.
	%
	% The basepath varibel can be used to specifiy the path of the excel file
	% to use with this script, while fileName is the name of the excel file.
	
	%basePath = "C:\Users\mwillig\Nextcloud\PhD-Pairmotion\" + ...
	%           "SurveyOverviewOSpaper\DSE, Design Space Exploration\" + ...
	%           "DSE for NoC (mit Julian,Cornelia)\Matlab";
	
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
	sheetNum = 3;
    plot_title = ""
	switch sheetNum
	    case 1
            pktLength = transpose(load(dbPath+'Bitcomplement.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Bitcomplement.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Bitcomplement.mat','timePerFlit').timePerFlit);
            plot_title = "Bitcomplement"
	    case 2
            pktLength = transpose(load(dbPath+'Bitshuffle.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Bitshuffle.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Bitshuffle.mat','timePerFlit').timePerFlit);
            plot_title = "Bitshuffle"
	    case 3
            pktLength = transpose(load(dbPath+'Bitrotate.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Bitrotate.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Bitrotate.mat','timePerFlit').timePerFlit);
            plot_title = "Bitrotate" 
	    case 4
            pktLength = transpose(load(dbPath+'Bitrevers.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Bitrevers.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Bitrevers.mat','timePerFlit').timePerFlit);
            plot_title = "Bitrevers"
	    case 5
            pktLength = transpose(load(dbPath+'Transpose1.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Transpose1.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Transpose1.mat','timePerFlit').timePerFlit);
            plot_title = "Transpose1"
	    case 6
            pktLength = transpose(load(dbPath+'Transpose2.mat','packetLength').packetLength);
            buffSizePy = transpose(load(dbPath+'Transpose2.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load(dbPath+'Transpose2.mat','timePerFlit').timePerFlit);
            plot_title = "Transpose2"
	    otherwise
            pktLength = transpose(load('Bitcomplement.mat','packetLength').packetLength);
            buffSizePy = transpose(load('Bitcomplement.mat','bufferSize').bufferSize);
            timePerFlitPy = transpose(load('Bitcomplement.mat','timePerFlit').timePerFlit);       
	end

	% Same goes for the column names. Matlab identifies columns by name. column
	% names must be stated in the same order as given by the excel file.
	% opts.VariableNames = ["buffersize", "packetlength", "injectionrate",  ...
	                    %   "avglat", "minlat", "maxlat", "timeperflit"];
	% each colmun needs a datatype to be converted to when creating an internal
	% represenation of the data
	% opts.VariableTypes = ["double", "double", "double", "int32", ...
	%                       "int32", "int32", "double"];
	
	% Using all the settings done above, we now finally load the tables data to
	% matlab. 
	% NoCtable = readtable(tableSearchPath, opts, "UseExcel", false);
	
	% Clear loading settings to free memory
	% clear opts
	
	
	%  ------------            Interpolate data grid         -----------------
	% store data from excel files in variables for easier access
	% buffersize = NoCtable.buffersize;
	% timeperflit = NoCtable.timeperflit;
	% packetlength = NoCtable.packetlength;
	
	% specify a meshgrid, which means that arrays xq and yq are created. X
	% values are stored in xq and Y values are stored in yq. Both arrays are
	% the same length and accessing and contain compared all combinations of
	% the grid. The values range are defined by the arguments of meshgrid in
	% the way of intervallStart:IncrementPerStep:Steps
	[xq, yq] = meshgrid(0:1:600, 0:1:600);
	
	% the next step tries to interpolate data for the whole grid of xq and yq
	% by using the measurement data. For interpolation, cubiy splines are used
	
	% @TODO: maybe use 'v4', eventually compare different ones
	%vq = griddata(buffersize, packetlength, timeperflit, xq, yq, 'cubic');
	% the mesh is plotted.
	
    vq = griddata(buffSizePy, pktLength, timePerFlitPy, xq, yq, 'cubic');
    mesh_plot = mesh(xq,yq,vq);
	hold on;

    %pktLength_equal = isequal(pktLength, packetlength);
    %buffSize_equal = isequal(buffSizePy, buffersize);

    %disp(pktLength_equal);
    %disp(buffSize_equal);
    

	% now the measured data are added to the plot.
	%plot3_dse = plot3(buffersize,packetlength,timeperflit,'o');
	plot3_dse = plot3(buffSizePy,pktLength, timePerFlitPy, 'o');
	
	%  ------------                plot settings            -----------------
	% The following lines are used to do several settings to the graph.
	% Create zlabel
	zlabel({'avg. latency'});
	% Create ylabel
	ylabel({'packet length'});
	% Create xlabel
	xlabel({'buffer size'});
	% Create title
	title({'PANACA NoC Simulator ('+plot_title+')'});

	% adds a colorbar as legend to the side of the plot
	colorbar();
	grid('on');
	hold('on');
	
	
	%  ------------              Finding the minimum         -----------------
	% We try to find the global minimum in the interpolated data. If we find a 
	% minimum, we want to use a data tip to highlight it's position in the
	% graph. The global interpolated minimum is our suggestion for a new
	% measurement
	
	% find the global minimum in the interpolated data
	min_timeperflit = min(min(vq));
	% finde the indices of the minimum in the table of interpolated data. The
	% indices are the same as in the x and y data arrays.
	[row, col] = find(vq==min_timeperflit);
    

	
    row0 = row(1);
    col0 = col(1);
 
	% find the minimum x and y data by using the indices.
	min_buffersize = xq(row0, col0);
	min_packetlength = yq(row0, col0);
	
	% We must prepare our data tip, before we can used it to display data
	% set a template for data tip axis names
	% x axis data tip display name
	mesh_plot.DataTipTemplate.DataTipRows(1).Label = 'buffer size:';
	% y axis data tip display name
	mesh_plot.DataTipTemplate.DataTipRows(2).Label = 'packet length:';
	% z axis data tip display name
	mesh_plot.DataTipTemplate.DataTipRows(3).Label = 'avg latency per flit:';
	
	% create a data tip at the suspected minimum
	% the following if clause is a work around to fix the case when the minimum
	% is reached on multiple settings. In this case, we will only ceate a data
	% tip on the first coordinates of the minimum in the list of coordinates
	if length(min_packetlength) > 1
	    min_packetlength = min_packetlength(1, 1);
	    min_timeperflit =  min_timeperflit(1, 1);
	end
	
	% find the index of the minimum in the vq data
	minIndex = find(vq==min_timeperflit);
    %disp(min_timeperflit)
	% create a data tip by using the index of the calculated minimum
	datatip(mesh_plot, 'DataIndex', minIndex(1));
	
	% debugging output i guess? Will print multiple lines if the minimum is
	% reached on multiple coordinates
	fprintf("min_timeperflit: %8.5f min_buffersize: %8.3f" + ...
	        " min_packetlength: %8.3f \n", ...
	        min_timeperflit, min_buffersize, min_packetlength);
	%disp(length(min_packetlength));
	%disp(min_timeperflit);
	%disp(length(min_buffersize));
	% we are finished!

