    
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
	sheetNum = 1;
    plot_title = ""
    pktLength = transpose(load(dbPath+'Bitcomplement.mat','packetLength').packetLength);
    buffSizePy = transpose(load(dbPath+'Bitcomplement.mat','bufferSize').bufferSize);
    timePerFlitPy = transpose(load(dbPath+'Bitcomplement.mat','timePerFlit').timePerFlit);
    plot_title = "Bitcomplement"
    % create random field with autocorrelation
    [X] = meshgrid(0:1:400);
    [Y]= meshgrid(0:1:400);
    Z = randn(size(X));
    Z = imfilter(Z,fspecial('gaussian',[40 40],8));

    % sample the field
    n = 128;
    %x = rand(n,1)*500;
    x= pktLength
    %y = rand(n,1)*500;
    y = buffSizePy
    %z = interp2(X,Y,Z,x,y);
    vq = griddata(timePerFlitPy, pktLength, buffSizePy, X, Y, 'cubic');
    z = timePerFlitPy

    % plot the random field
    subplot(2,2,1)
    imagesc(X(1,:),Y(:,1),Z); axis image; axis xy
    hold on
    plot(x,y,'.k')
    title('random field with sampling locations')

    % calculate the sample variogram 
    v = variogram([x y],vq,'plotit',false,'maxdist',100);
    % and fit a spherical variogram
    subplot(2,2,2)
    [dum,dum,dum,vstruct] = variogramfit(v.distance,v.val,[],[],[],'model','stable');
    title('variogram')

    % now use the sampled locations in a kriging
    [Zhat,Zvar] = kriging(vstruct,x,y,z,X,Y);
    subplot(2,2,3)
    imagesc(X(1,:),Y(:,1),Zhat); axis image; axis xy
    title('kriging predictions')
    subplot(2,2,4)
    contour(X,Y,Zvar); axis image
    title('kriging variance')

function [zi,s2zi] = kriging(vstruct,x,y,z,xi,yi,chunksize)
% interpolation with ordinary kriging in two dimensions
%
% Syntax:
%
%     [zi,zivar] = kriging(vstruct,x,y,z,xi,yi)
%     [zi,zivar] = kriging(vstruct,x,y,z,xi,yi,chunksize)
%
% Description:
%
%     kriging uses ordinary kriging to interpolate a variable z measured at
%     locations with the coordinates x and y at unsampled locations xi, yi.
%     The function requires the variable vstruct that contains all
%     necessary information on the variogram. vstruct is the forth output
%     argument of the function variogramfit.
%
%     This is a rudimentary, but easy to use function to perform a simple
%     kriging interpolation. I call it rudimentary since it always includes
%     ALL observations to estimate values at unsampled locations. This may
%     not be necessary when sample locations are not within the
%     autocorrelation range but would require something like a k nearest
%     neighbor search algorithm or something similar. Thus, the algorithms
%     works best for relatively small numbers of observations (100-500).
%     For larger numbers of observations I recommend the use of GSTAT.
%
%     Note that kriging fails if there are two or more observations at one
%     location or very, very close to each other. This may cause that the 
%     system of equation is badly conditioned. Currently, I use the
%     pseudo-inverse (pinv) to come around this problem. If you have better
%     ideas, please let me know.
%
% Input arguments:
%
%     vstruct   structure array with variogram information as returned
%               variogramfit (forth output argument)
%     x,y       coordinates of observations
%     z         values of observations
%     xi,yi     coordinates of locations for predictions 
%     chunksize nr of elements in zi that are processed at one time.
%               The default is 100, but this depends largely on your 
%               available main memory and numel(x).
%
% Output arguments:
%
%     zi        kriging predictions
%     zivar     kriging variance
%
% Example:
%
%     % create random field with autocorrelation
%     [X,Y] = meshgrid(0:500);
%     Z = randn(size(X));
%     Z = imfilter(Z,fspecial('gaussian',[40 40],8));
%
%     % sample the field
%     n = 500;
%     x = rand(n,1)*500;
%     y = rand(n,1)*500;
%     z = interp2(X,Y,Z,x,y);
%
%     % plot the random field
%     subplot(2,2,1)
%     imagesc(X(1,:),Y(:,1),Z); axis image; axis xy
%     hold on
%     plot(x,y,'.k')
%     title('random field with sampling locations')
%
%     % calculate the sample variogram
%     v = variogram([x y],z,'plotit',false,'maxdist',100);
%     % and fit a spherical variogram
%     subplot(2,2,2)
%     [dum,dum,dum,vstruct] = variogramfit(v.distance,v.val,[],[],[],'model','stable');
%     title('variogram')
%
%     % now use the sampled locations in a kriging
%     [Zhat,Zvar] = kriging(vstruct,x,y,z,X,Y);
%     subplot(2,2,3)
%     imagesc(X(1,:),Y(:,1),Zhat); axis image; axis xy
%     title('kriging predictions')
%     subplot(2,2,4)
%     contour(X,Y,Zvar); axis image
%     title('kriging variance')
%
%
% see also: variogram, variogramfit, consolidator, pinv
%
% Date: 13. October, 2010
% Author: Wolfgang Schwanghart (w.schwanghart[at]unibas.ch)
% size of input arguments
sizest = size(xi);
numest = numel(xi);
numobs = numel(x);
% force column vectors
xi = xi(:);
yi = yi(:);
x  = x(:);
y  = y(:);
z  = z(:);
if nargin == 6;
    chunksize = 100;
elseif nargin == 7;
else
    error('wrong number of input arguments')
end
% check if the latest version of variogramfit is used
if ~isfield(vstruct, 'func')
    error('please download the latest version of variogramfit from the FEX')
end
% variogram function definitions
switch lower(vstruct.model)    
    case {'whittle' 'matern'}
        error('whittle and matern are not supported yet');
    case 'stable'
        stablealpha = vstruct.stablealpha; %#ok<NASGU> % will be used in an anonymous function
end
% distance matrix of locations with known values
Dx = hypot(bsxfun(@minus,x,x'),bsxfun(@minus,y,y'));
% if we have a bounded variogram model, it is convenient to set distances
% that are longer than the range to the range since from here on the
% variogram value remains the same and we don£t need composite functions.
switch vstruct.type;
    case 'bounded'
        Dx = min(Dx,vstruct.range);
    otherwise
end
% now calculate the matrix with variogram values 
A = vstruct.func([vstruct.range vstruct.sill],Dx);
if ~isempty(vstruct.nugget)
    A = A+vstruct.nugget;
end
% the matrix must be expanded by one line and one row to account for
% condition, that all weights must sum to one (lagrange multiplier)
A = [[A ones(numobs,1)];ones(1,numobs) 0];
% A is often very badly conditioned. Hence we use the Pseudo-Inverse for
% solving the equations
A = pinv(A);
% we also need to expand z
z  = [z;0];
% allocate the output zi
zi = nan(numest,1);
if nargout == 2;
    s2zi = nan(numest,1);
    krigvariance = true;
else
    krigvariance = false;
end
% parametrize engine
nrloops   = ceil(numest/chunksize);
% initialize the waitbar
h  = waitbar(0,'Kr...kr...kriging');
% now loop 
for r = 1:nrloops;
    % waitbar 
    waitbar(r / nrloops,h);
    
    % built chunks
    if r<nrloops
        IX = (r-1)*chunksize +1 : r*chunksize;
    else
        IX = (r-1)*chunksize +1 : numest;
        chunksize = numel(IX);
    end
    
    % build b
    b = hypot(bsxfun(@minus,x,xi(IX)'),bsxfun(@minus,y,yi(IX)'));
    % again set maximum distances to the range
    switch vstruct.type
        case 'bounded'
            b = min(vstruct.range,b);
    end
    
    % expand b with ones
    b = [vstruct.func([vstruct.range vstruct.sill],b);ones(1,chunksize)];
    if ~isempty(vstruct.nugget)
        b = b+vstruct.nugget;
    end
    
    % solve system
    lambda = A*b;
    
    % estimate zi
    zi(IX)  = lambda'*z;
    
    % calculate kriging variance
    if krigvariance
        s2zi(IX) = sum(b.*lambda,1);
    end
    
end
% close waitbar
close(h)
% reshape zi
zi = reshape(zi,sizest);
if krigvariance
    s2zi = reshape(s2zi,sizest);
end
end


%Variogram
function S = variogram(x,y,varargin)
% isotropic and anisotropic experimental (semi-)variogram
%
% Syntax:
%   d = variogram(x,y)
%   d = variogram(x,y,'propertyname','propertyvalue',...)
%
% Description:
%   variogram calculates the experimental variogram in various 
%   dimensions. 
%
% Input:
%   x - array with coordinates. Each row is a location in a 
%       size(x,2)-dimensional space (e.g. [x y elevation])
%   y - column vector with values of the locations in x. 
%
% Propertyname/-value pairs:
%   nrbins - number bins the distance should be grouped into
%            (default = 20)
%   maxdist - maximum distance for variogram calculation
%            (default = maximum distance in the dataset / 2)
%   type -   'gamma' returns the variogram value (default)
%            'cloud1' returns the binned variogram cloud
%            'cloud2' returns the variogram cloud
%   plotit - true -> plot variogram
%            false -> don't plot (default)
%   subsample - number of randomly drawn points if large datasets are used.
%               scalar (positive integer, e.g. 3000)
%               inf (default) = no subsampling
%   anisotropy - false (default), true (works only in two dimensions)
%   thetastep - if anisotropy is set to true, specifying thetastep 
%            allows you the angle width (default 30°)
%   
%   
% Output:
%   d - structure array with distance and gamma - vector
%   
% Example: Generate a random field with periodic variation in x direction
% 
%     x = rand(1000,1)*4-2;  
%     y = rand(1000,1)*4-2;
%     z = 3*sin(x*15)+ randn(size(x));
%
%     subplot(2,2,1)
%     scatter(x,y,4,z,'filled'); box on;
%     ylabel('y'); xlabel('x')
%     title('data (coloring according to z-value)')
%     subplot(2,2,2)
%     hist(z,20)
%     ylabel('frequency'); xlabel('z')
%     title('histogram of z-values')
%     subplot(2,2,3)
%     d = variogram([x y],z,'plotit',true,'nrbins',50);
%     title('Isotropic variogram')
%     subplot(2,2,4)
%     d2 = variogram([x y],z,'plotit',true,'nrbins',50,'anisotropy',true);
%     title('Anisotropic variogram')
%
% Requirements:
%   The function uses parseargs (objectId=10670) 
%   by Malcolm wood as subfunction.
%
% See also: KRIGING, VARIOGRAMFIT
%
% Date: 9. January, 2013
% Author: Wolfgang Schwanghart
% error checking
if size(y,1) ~= size(x,1);
    error('x and y must have the same number of rows')
end
% check for nans
II = any(isnan(x),2) | isnan(y);
x(II,:) = [];
y(II)   = [];
% extent of dataset
minx = min(x,[],1);
maxx = max(x,[],1);
maxd = sqrt(sum((maxx-minx).^2));
nrdims = size(x,2);
% check input using PARSEARGS
params.nrbins      = 20;
params.maxdist     = maxd/2;
params.type        = {'default','gamma','cloud1','cloud2'};
params.plotit      = false;
params.anisotropy  = false;
params.thetastep   = 30;
params.subsample   = inf;
params = parseargs(params,varargin{:});
if params.maxdist > maxd;
    warning('Matlab:Variogram',...
            ['Maximum distance exceeds maximum distance \n' ... 
             'in the dataset. maxdist was decreased to ' num2str(maxd) ]);
    params.maxdist  = maxd;
end
if params.anisotropy && nrdims ~= 2 
    params.anisotropy = false;
    warning('Matlab:Variogram',...
            'Anistropy is only supported for 2D data');
end
% take only a subset of the data;
if ~isinf(params.subsample) && numel(y)>params.subsample;
    IX = randperm(numel(y),params.subsample);
    x  = x(IX,:);
    y  = y(IX,:);
end
% calculate bin tolerance
tol = params.maxdist/params.nrbins;
% calculate distance matrix
iid = distmat(x,params.maxdist);
% calculate squared difference between values of coordinate pairs
lam      = (y(iid(:,1))-y(iid(:,2))).^2;
%
params.thetastep = params.thetastep;
% anisotropy
if params.anisotropy 
    nrthetaedges = floor(180/(params.thetastep))+1;
  
    % calculate with radians, not degrees
    params.thetastep = params.thetastep/180*pi;
    % calculate angles, note that angle is calculated clockwise from top
    theta    = atan2(x(iid(:,2),1)-x(iid(:,1),1),...
                     x(iid(:,2),2)-x(iid(:,1),2));
    
    % only the semicircle is necessary for the directions
    I        = theta < 0;
    theta(I) = theta(I)+pi;
    I        = theta >= pi-params.thetastep/2;
    theta(I) = 0;
        
    % create a vector with edges for binning of theta
    % directions go from 0 to 180 degrees;
    thetaedges = linspace(-params.thetastep/2,pi-params.thetastep/2,nrthetaedges);
    
    % bin theta
    [ntheta,ixtheta] = histc(theta,thetaedges);
    
    % bin centers
    thetacents = thetaedges(1:end)+params.thetastep/2;
    thetacents(end) = pi; %[];
end
% calculate variogram
switch params.type
    case {'default','gamma'}
        % variogram anonymous function
        fvar     = @(x) 1./(2*numel(x)) * sum(x);
        
        % distance bins
        edges      = linspace(0,params.maxdist,params.nrbins+1);
        edges(end) = inf;
        [nedge,ixedge] = histc(iid(:,3),edges);
        
        if params.anisotropy
            S.val      = accumarray([ixedge ixtheta],lam,...
                                 [numel(edges) numel(thetaedges)],fvar,nan);
            S.val(:,end)=S.val(:,1); 
            S.theta    = thetacents;
            S.num      = accumarray([ixedge ixtheta],ones(size(lam)),...
                                 [numel(edges) numel(thetaedges)],@sum,nan);
            S.num(:,end)=S.num(:,1);                 
        else
            S.val      = accumarray(ixedge,lam,[numel(edges) 1],fvar,nan);     
            S.num      = accumarray(ixedge,ones(size(lam)),[numel(edges) 1],@sum,nan);
        end
        S.distance = (edges(1:end-1)+tol/2)';
        S.val(end,:) = [];
        S.num(end,:) = [];
    case 'cloud1'
        edges      = linspace(0,params.maxdist,params.nrbins+1);
        edges(end) = inf;
        
        [nedge,ixedge] = histc(iid(:,3),edges);
        
        S.distance = edges(ixedge) + tol/2;
        S.distance = S.distance(:);
        S.val      = lam;  
        if params.anisotropy            
            S.theta   = thetacents(ixtheta);
        end
    case 'cloud2'
        S.distance = iid(:,3);
        S.val      = lam;
        if params.anisotropy            
            S.theta   = thetacents(ixtheta);
        end
end
% create plot if desired
if params.plotit
    switch params.type
        case {'default','gamma'}
            marker = 'o--';
        otherwise
            marker = '.';
    end
    
    if ~params.anisotropy
        plot(S.distance,S.val,marker);
        axis([0 params.maxdist 0 max(S.val)*1.1]);
        xlabel('h');
        ylabel('\gamma (h)');
        title('(Semi-)Variogram');
    else
        [Xi,Yi] = pol2cart(repmat(S.theta,numel(S.distance),1),repmat(S.distance,1,numel(S.theta)));
        surf(Xi,Yi,S.val)
        xlabel('h y-direction')
        ylabel('h x-direction')
        zlabel('\gamma (h)')
        title('directional variogram')
%         set(gca,'DataAspectRatio',[1 1 1/30])
    end
end
        
end
% subfunction distmat
function iid = distmat(X,dmax)
% constrained distance function
%
% iid -> [rows, columns, distance]
 
n     = size(X,1);
nrdim = size(X,2);
if size(X,1) < 1000;
    [i,j] = find(triu(true(n)));
    if nrdim == 1;
        d = abs(X(i)-X(j));
    elseif nrdim == 2;
        d = hypot(X(i,1)-X(j,1),X(i,2)-X(j,2));
    else
        d = sqrt(sum((X(i,:)-X(j,:)).^2,2));
    end
    I = d<=dmax;
    iid = [i(I) j(I) d(I)];
else
    ix = (1:n)';
    if nrdim == 1;
        iid = arrayfun(@distmatsub1d,(1:n)','UniformOutput',false);
    elseif nrdim == 2;
        % if needed change distmatsub to distmatsub2d which is numerically
        % better but slower
        iid = arrayfun(@distmatsub,(1:n)','UniformOutput',false);
    else
        iid = arrayfun(@distmatsub,(1:n)','UniformOutput',false);
    end
    nn  = cellfun(@(x) size(x,1),iid,'UniformOutput',true);  
    I   = nn>0;
    ix  = ix(I);
    nn  = nn(I);
    nncum = cumsum(nn);
    c     = zeros(nncum(end),1);
    c([1;nncum(1:end-1)+1]) = 1;
    i = ix(cumsum(c));
    iid = [i cell2mat(iid)];
    
end
function iid = distmatsub1d(i) 
    j  = (i+1:n)'; 
    d  = abs(X(i)-X(j));
    I  = d<=dmax;
    iid = [j(I) d(I)];
end
function iid = distmatsub2d(i)  %#ok<DEFNU>
    j  = (i+1:n)'; 
    d = hypot(X(i,1) - X(j,1),X(i,2) - X(j,2));
    I  = d<=dmax;
    iid = [j(I) d(I)];
end
    
function iid = distmatsub(i)
    j  = (i+1:n)'; 
    d = sqrt(sum(bsxfun(@minus,X(i,:),X(j,:)).^2,2));
    I  = d<=dmax;
    iid = [j(I) d(I)];
end
end
% subfunction parseargs


% Variogramfit
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,c,n,S] = variogramfit(h,gammaexp,a0,c0,numobs,varargin)
% fit a theoretical variogram to an experimental variogram
%
% Syntax:
%
%     [a,c,n] = variogramfit(h,gammaexp,a0,c0)
%     [a,c,n] = variogramfit(h,gammaexp,a0,c0,numobs)
%     [a,c,n] = variogramfit(h,gammaexp,a0,c0,numobs,'pn','pv',...)
%     [a,c,n,S] = variogramfit(...)
%
% Description:
%
%     variogramfit performs a least squares fit of various theoretical 
%     variograms to an experimental, isotropic variogram. The user can
%     choose between various bounded (e.g. spherical) and unbounded (e.g.
%     exponential) models. A nugget variance can be modelled as well, but
%     higher nested models are not supported.
%
%     The function works best with the function fminsearchbnd available on
%     the FEX. You should download it from the File Exchange (File ID:
%     #8277). If you don't have fminsearchbnd, variogramfit uses
%     fminsearch. The problem with fminsearch is, that it might return 
%     negative variances or ranges.
%
%     The variogram fitting algorithm is in particular sensitive to initial
%     values below the optimal solution. In case you have no idea of
%     initial values variogramfit calculates initial values for you
%     (c0 = max(gammaexp); a0 = max(h)*2/3;). If this is a reasonable
%     guess remains to be answered. Hence, visually inspecting your data
%     and estimating a theoretical variogram by hand should always be
%     your first choice.
%
%     Note that for unbounded models, the supplied parameter a0 (range) is
%     the distance where gamma equals 95% of the sill variance. The
%     returned parameter a0, however, is the parameter r in the model. The
%     range at 95% of the sill variance is then approximately 3*r.
%
% Input arguments:
%
%     h         lag distance of the experimental variogram
%     gammaexp  experimental variogram values (gamma)
%     a0        initial value (scalar) for range
%     c0        initial value (scalar) for sill variance
%     numobs    number of observations per lag distance (used for weight
%               function)
%
% Output arguments:
%
%     a         range
%     c         sill
%     n         nugget (empty if nugget variance is not applied)
%     S         structure array with additional information
%               .range
%               .sill
%               .nugget
%               .model - theoretical variogram
%               .func - anonymous function of variogram model (only the
%               function within range for bounded models)
%               .h  - distance
%               .gamma  - experimental variogram values
%               .gammahat - estimated variogram values
%               .residuals - residuals
%               .Rs - R-square of goodness of fit
%               .weights - weights
%               .exitflag - see fminsearch
%               .algorithm - see fminsearch
%               .funcCount - see fminsearch
%               .iterations - see fminsearch
%               .message - see fminsearch
%
% Property name/property values:
% 
%     'model'   a string that defines the function that can be fitted 
%               to the experimental variogram. 
% 
%               Supported bounded functions are:
%               'blinear' (bounded linear) 
%               'circular' (circular model)
%               'spherical' (spherical model, =default)
%               'pentaspherical' (pentaspherical model)
% 
%               Supported unbounded functions are:
%               'exponential' (exponential model)
%               'gaussian' (gaussian variogram)
%               'whittle' Whittle's elementary correlation (involves a
%                         modified Bessel function of the second kind.
%               'stable' (stable models sensu Wackernagel 1995). Same as
%                         gaussian, but with different exponents. Supply 
%                         the exponent alpha (<2) in an additional pn,pv 
%                         pair: 
%                        'stablealpha',alpha (default = 1.5).
%               'matern' Matern model. Requires an additional pn,pv pair. 
%                        'nu',nu (shape parameter > 0, default = 1)
%                        Note that for particular values of nu the matern 
%                        model reduces to other authorized variogram models.
%                        nu = 0.5 : exponential model
%                        nu = 1 : Whittles model
%                        nu -> inf : Gaussian model
%               
%               See Webster and Oliver (2001) for an overview on variogram 
%               models. See Minasny & McBratney (2005) for an introduction
%               to the Matern variogram.
%           
%     'nugget'  initial value (scalar) for nugget variance. The default
%               value is []. In this case variogramfit doesn't fit a nugget
%               variance. 
% 
%     'plotit'  true (default), false: plot experimental and theoretical 
%               variogram together.
% 
%     'solver'  'fminsearchbnd' (default) same as fminsearch , but with  
%               bound constraints by transformation (function by John 
%               D'Errico, File ID: #8277 on the FEX). The advantage in 
%               applying fminsearchbnd is that upper and lower bound 
%               constraints can be applied. That prevents that nugget 
%               variance or range may become negative.           
%               'fminsearch'
%
%     'weightfun' 'none' (default). 'cressie85' and 'mcbratney86' require
%               you to include the number of observations per experimental
%               gamma value (as returned by VARIOGRAM). 
%               'cressie85' uses m(hi)/gammahat(hi)^2 as weights
%               'mcbratney86' uses m(hi)*gammaexp(hi)/gammahat(hi)^3
%               
%
% Example: fit a variogram to experimental data
%
%     load variogramexample
%     a0 = 15; % initial value: range 
%     c0 = 0.1; % initial value: sill 
%     [a,c,n] = variogramfit(h,gammaexp,a0,c0,[],...
%                            'solver','fminsearchbnd',...
%                            'nugget',0,...
%                            'plotit',true);
%
%           
% See also: VARIOGRAM, FMINSEARCH, FMINSEARCHBND
%           
%
% References: Wackernagel, H. (1995): Multivariate Geostatistics, Springer.
%             Webster, R., Oliver, M. (2001): Geostatistics for
%             Environmental Scientists. Wiley & Sons.
%             Minsasny, B., McBratney, A. B. (2005): The Matérn function as
%             general model for soil variograms. Geoderma, 3-4, 192-207.
% 
% Date: 7. October, 2010
% Author: Wolfgang Schwanghart (w.schwanghart[at]unibas.ch)
% check input arguments
if nargin == 0
    help variogramfit
    return
elseif nargin>0 && nargin < 2;
    error('Variogramfit:inputargs',...
          'wrong number of input arguments');
end
if ~exist('a0','var') || isempty(a0)
    a0 = max(h)*2/3;
end
if ~exist('c0','var') || isempty(c0)
    c0 = max(gammaexp);
end
if ~exist('numobs','var') || isempty(a0)
    numobs = [];
end
      
% check input parameters
params.model       = 'spherical';
params.nugget      = [];
params.plotit      = true;
params.solver      = {'fminsearchbnd','fminsearch'};
params.stablealpha = 1.5;
params.weightfun   = {'none','cressie85','mcbratney86'};
params.nu          = 1;
params = parseargs(params,varargin{:});
% check if fminsearchbnd is in the search path
switch lower(params.solver)
    case 'fminsearchbnd'
        if ~exist('fminsearchbnd.m','file')==2
            params.solver = 'fminsearch';
            warning('Variogramfit:fminsearchbnd',...
            'fminsearchbnd was not found. fminsearch is used instead')
        end
end
% check if h and gammaexp are vectors and have the same size
if ~isvector(h) || ~isvector(gammaexp)
    error('Variogramfit:inputargs',...
          'h and gammaexp must be vectors');
end
% force column vectors
h = h(:);
gammaexp = gammaexp(:);
% check size of supplied vectors 
if numel(h) ~= numel(gammaexp)
    error('Variogramfit:inputargs',...
          'h and gammaexp must have same size');
end
% remove nans;
nans = isnan(h) | isnan(gammaexp);
if any(nans);
    h(nans) = [];
    gammaexp(nans) = [];
    if ~isempty(numobs)
        numobs(nans) = [];
    end
end
% check weight inputs
if isempty(numobs);
    params.weightfun = 'none';
end
    
% create options for fminsearch
options = optimset('MaxFunEvals',1000000);
% create vector with initial values
% b(1) range
% b(2) sill
% b(3) nugget if supplied
b0 = [a0 c0 params.nugget];
% variogram function definitions
switch lower(params.model)    
    case 'spherical'
        type = 'bounded';
        func = @(b,h)b(2)*((3*h./(2*b(1)))-1/2*(h./b(1)).^3);
    case 'pentaspherical'
        type = 'bounded';
        func = @(b,h)b(2)*(15*h./(8*b(1))-5/4*(h./b(1)).^3+3/8*(h./b(1)).^5);
    case 'blinear'
        type = 'bounded';
        func = @(b,h)b(2)*(h./b(1));
    case 'circular'
        type = 'bounded';
        func = @(b,h)b(2)*(1-(2./pi)*acos(h./b(1))+2*h/(pi*b(1)).*sqrt(1-(h.^2)/(b(1)^2)));
    case 'exponential'
        type = 'unbounded';
        func = @(b,h)b(2)*(1-exp(-h./b(1)));
    case 'gaussian'
        type = 'unbounded';
        func = @(b,h)b(2)*(1-exp(-(h.^2)/(b(1)^2)));
    case 'stable'
        type = 'unbounded';
        stablealpha = params.stablealpha;
        func = @(b,h)b(2)*(1-exp(-(h.^stablealpha)/(b(1)^stablealpha)));  
    case 'whittle'
        type = 'unbounded';
        func = @(b,h)b(2)*(1-h/b(1).*besselk(1,h/b(1)));
    case 'matern'
        type = 'unbounded';
        func = @(b,h)b(2)*(1-(1/((2^(params.nu-1))*gamma(params.nu))) * (h/b(1)).^params.nu .* besselk(params.nu,h/b(1)));
    otherwise
        error('unknown model')
end
% check if there are zero distances 
% if yes, remove them, since the besselk function returns nan for
% zero
switch lower(params.model) 
    case {'whittle','matern'}
        izero = h==0;
        if any(izero)
            flagzerodistances = true;
        else
            flagzerodistances = false;
        end
    otherwise
        flagzerodistances = false;
end
        
% if model type is unbounded, then the parameter b(1) is r, which is
% approximately range/3. 
switch type
    case 'unbounded'
        b0(1) = b0(1)/3;
end
% nugget variance
if isempty(params.nugget)
    nugget = false;
    funnugget = @(b) 0;
else
    nugget = true;
    funnugget = @(b) b(3);
end
% generate upper and lower bounds when fminsearchbnd is used
switch lower(params.solver)
    case {'fminsearchbnd'};
        % lower bounds
        lb = zeros(size(b0));
        % upper bounds
        if nugget;
            ub = [inf max(gammaexp) max(gammaexp)]; %
        else
            ub = [inf max(gammaexp)];
        end
end
% create weights (see Webster and Oliver)
switch params.weightfun
    case 'cressie85'
        weights = @(b,h) (numobs./variofun(b,h).^2)./sum(numobs./variofun(b,h).^2);
    case 'mcbratney86'
        weights = @(b,h) (numobs.*gammaexp./variofun(b,h).^3)/sum(numobs.*gammaexp./variofun(b,h).^3);
    otherwise
        weights = @(b,h) 1;
end
% create objective function: weighted least square
objectfun = @(b)sum(((variofun(b,h)-gammaexp).^2).*weights(b,h));
% call solver
switch lower(params.solver)
    case 'fminsearch'                
        % call fminsearch
        [b,fval,exitflag,output] = fminsearch(objectfun,b0,options);
    case 'fminsearchbnd'
        % call fminsearchbnd
        [b,fval,exitflag,output] = fminsearchbnd(objectfun,b0,lb,ub,options);
    otherwise
        error('Variogramfit:Solver','unknown or unsupported solver')
end
% prepare output
a = b(1); %range
c = b(2); %sill
if nugget;
    n = b(3);%nugget
else
    n = [];
end
% Create structure array with results 
if nargout == 4;    
    S.model     = lower(params.model); % model
    S.func      = func;
    S.type      = type;
    switch S.model 
        case 'matern';
            S.nu = params.nu;
        case 'stable';
            S.stablealpha = params.stablealpha;
    end
        
    
    S.range     = a;
    S.sill      = c;
    S.nugget    = n;
    S.h         = h; % distance
    S.gamma     = gammaexp; % experimental values
    S.gammahat  = variofun(b,h); % estimated values
    S.residuals = gammaexp-S.gammahat; % residuals
    COVyhaty    = cov(S.gammahat,gammaexp);
    S.Rs        = (COVyhaty(2).^2) ./...
                  (var(S.gammahat).*var(gammaexp)); % Rsquare
    S.weights   = weights(b,h); %weights
    S.weightfun = params.weightfun;
    S.exitflag  = exitflag; % exitflag (see doc fminsearch)
    S.algorithm = output.algorithm;
    S.funcCount = output.funcCount;
    S.iterations= output.iterations;
    S.message   = output.message;
end
% if you want to plot the results...
if params.plotit
    switch lower(type)
        case 'bounded'
            plot(h,gammaexp,'rs','MarkerSize',10);
            hold on
            fplot(@(h) funnugget(b) + func(b,h),[0 b(1)])
            fplot(@(h) funnugget(b) + b(2),[b(1) max(h)])
            
        case 'unbounded'
            plot(h,gammaexp,'rs','MarkerSize',10);
            hold on
            fplot(@(h) funnugget(b) + func(b,h),[0 max(h)])
    end
    axis([0 max(h) 0 max(gammaexp)])
    xlabel('lag distance h')
    ylabel('\gamma(h)')
    hold off
end
% fitting functions for  fminsearch/bnd
function gammahat = variofun(b,h)
    
    switch type
        % bounded model
        case 'bounded'
            I = h<=b(1);
            gammahat     = zeros(size(I));
            gammahat(I)  = funnugget(b) + func(b,h(I));
            gammahat(~I) = funnugget(b) + b(2);        
        % unbounded model
        case 'unbounded'
            gammahat = funnugget(b) + func(b,h);
            if flagzerodistances
                gammahat(izero) = funnugget(b);
            end    
    end
end
end
% and that's it...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseargs
function X = parseargs(X,varargin)
%PARSEARGS - Parses name-value pairs
%
% Behaves like setfield, but accepts multiple name-value pairs and provides
% some additional features:
% 1) If any field of X is an cell-array of strings, it can only be set to
%    one of those strings.  If no value is specified for that field, the
%    first string is selected.
% 2) Where the field is not empty, its data type cannot be changed
% 3) Where the field contains a scalar, its size cannot be changed.
%
% X = parseargs(X,name1,value1,name2,value2,...) 
%
% Intended for use as an argument parser for functions which multiple options.
% Example usage:
%
% function my_function(varargin)
%   X.StartValue = 0;
%   X.StopOnError = false;
%   X.SolverType = {'fixedstep','variablestep'};
%   X.OutputFile = 'out.txt';
%   X = parseargs(X,varargin{:});
%
% Then call (e.g.):
%
% my_function('OutputFile','out2.txt','SolverType','variablestep');
% The various #ok comments below are to stop MLint complaining about
% inefficient usage.  In all cases, the inefficient usage (of error, getfield, 
% setfield and find) is used to ensure compatibility with earlier versions
% of MATLAB.
remaining = nargin-1; % number of arguments other than X
count = 1;
fields = fieldnames(X);
modified = zeros(size(fields));
% Take input arguments two at a time until we run out.
while remaining>=2
    fieldname = varargin{count};
    fieldind = find(strcmp(fieldname,fields));
    if ~isempty(fieldind)
        oldvalue = getfield(X,fieldname); %#ok
        newvalue = varargin{count+1};
        if iscell(oldvalue)
            % Cell arrays must contain strings, and the new value must be
            % a string which appears in the list.
            if ~iscellstr(oldvalue)
                error(sprintf('All allowed values for "%s" must be strings',fieldname));  %#ok
            end
            if ~ischar(newvalue)
                error(sprintf('New value for "%s" must be a string',fieldname));  %#ok
            end
            if isempty(find(strcmp(oldvalue,newvalue))) %#ok
                error(sprintf('"%s" is not allowed for field "%s"',newvalue,fieldname));  %#ok
            end
        elseif ~isempty(oldvalue)
            % The caller isn't allowed to change the data type of a non-empty property,
            % and scalars must remain as scalars.
            if ~strcmp(class(oldvalue),class(newvalue))
                error(sprintf('Cannot change class of field "%s" from "%s" to "%s"',...
                    fieldname,class(oldvalue),class(newvalue))); %#ok
            elseif numel(oldvalue)==1 & numel(newvalue)~=1 %#ok
                error(sprintf('New value for "%s" must be a scalar',fieldname));  %#ok
            end
        end
        X = setfield(X,fieldname,newvalue); %#ok
        modified(fieldind) = 1;
    else
        error(['Not a valid field name: ' fieldname]);
    end
    remaining = remaining - 2;
    count = count + 2;
end
% Check that we had a value for every name.
if remaining~=0
    error('Odd number of arguments supplied.  Name-value pairs required');
end
% Now find cell arrays which were not modified by the above process, and select
% the first string.
notmodified = find(~modified);
for i=1:length(notmodified)
    fieldname = fields{notmodified(i)};
    oldvalue = getfield(X,fieldname); %#ok
    if iscell(oldvalue)
        if ~iscellstr(oldvalue)
            error(sprintf('All allowed values for "%s" must be strings',fieldname)); %#ok
        elseif isempty(oldvalue)
            error(sprintf('Empty cell array not allowed for field "%s"',fieldname)); %#ok
        end
        X = setfield(X,fieldname,oldvalue{1}); %#ok
    end
end
end


%%%%%%%% Fminsearchbnd

function [x,fval,exitflag,output] = fminsearchbnd(fun,x0,LB,UB,options,varargin)
% FMINSEARCHBND: FMINSEARCH, but with bound constraints by transformation
% usage: x=FMINSEARCHBND(fun,x0)
% usage: x=FMINSEARCHBND(fun,x0,LB)
% usage: x=FMINSEARCHBND(fun,x0,LB,UB)
% usage: x=FMINSEARCHBND(fun,x0,LB,UB,options)
% usage: x=FMINSEARCHBND(fun,x0,LB,UB,options,p1,p2,...)
% usage: [x,fval,exitflag,output]=FMINSEARCHBND(fun,x0,...)
% 
% arguments:
%  fun, x0, options - see the help for FMINSEARCH
%
%  LB - lower bound vector or array, must be the same size as x0
%
%       If no lower bounds exist for one of the variables, then
%       supply -inf for that variable.
%
%       If no lower bounds at all, then LB may be left empty.
%
%       Variables may be fixed in value by setting the corresponding
%       lower and upper bounds to exactly the same value.
%
%  UB - upper bound vector or array, must be the same size as x0
%
%       If no upper bounds exist for one of the variables, then
%       supply +inf for that variable.
%
%       If no upper bounds at all, then UB may be left empty.
%
%       Variables may be fixed in value by setting the corresponding
%       lower and upper bounds to exactly the same value.
%
% Notes:
%
%  If options is supplied, then TolX will apply to the transformed
%  variables. All other FMINSEARCH parameters should be unaffected.
%
%  Variables which are constrained by both a lower and an upper
%  bound will use a sin transformation. Those constrained by
%  only a lower or an upper bound will use a quadratic
%  transformation, and unconstrained variables will be left alone.
%
%  Variables may be fixed by setting their respective bounds equal.
%  In this case, the problem will be reduced in size for FMINSEARCH.
%
%  The bounds are inclusive inequalities, which admit the
%  boundary values themselves, but will not permit ANY function
%  evaluations outside the bounds. These constraints are strictly
%  followed.
%
%  If your problem has an EXCLUSIVE (strict) constraint which will
%  not admit evaluation at the bound itself, then you must provide
%  a slightly offset bound. An example of this is a function which
%  contains the log of one of its parameters. If you constrain the
%  variable to have a lower bound of zero, then FMINSEARCHBND may
%  try to evaluate the function exactly at zero.
%
%
% Example usage:
% rosen = @(x) (1-x(1)).^2 + 105*(x(2)-x(1).^2).^2;
%
% fminsearch(rosen,[3 3])     % unconstrained
% ans =
%    1.0000    1.0000
%
% fminsearchbnd(rosen,[3 3],[2 2],[])     % constrained
% ans =
%    2.0000    4.0000
%
% See test_main.m for other examples of use.
%
%
% See also: fminsearch, fminspleas
%
%
% Author: John D'Errico
% E-mail: woodchips@rochester.rr.com
% Release: 4
% Release date: 7/23/06

% size checks
xsize = size(x0);
x0 = x0(:);
n=length(x0);

if (nargin<3) || isempty(LB)
  LB = repmat(-inf,n,1);
else
  LB = LB(:);
end
if (nargin<4) || isempty(UB)
  UB = repmat(inf,n,1);
else
  UB = UB(:);
end

if (n~=length(LB)) || (n~=length(UB))
  error 'x0 is incompatible in size with either LB or UB.'
end

% set default options if necessary
if (nargin<5) || isempty(options)
  options = optimset('fminsearch');
end

% stuff into a struct to pass around
params.args = varargin;
params.LB = LB;
params.UB = UB;
params.fun = fun;
params.n = n;
% note that the number of parameters may actually vary if 
% a user has chosen to fix one or more parameters
params.xsize = xsize;
params.OutputFcn = [];

% 0 --> unconstrained variable
% 1 --> lower bound only
% 2 --> upper bound only
% 3 --> dual finite bounds
% 4 --> fixed variable
params.BoundClass = zeros(n,1);
for i=1:n
  k = isfinite(LB(i)) + 2*isfinite(UB(i));
  params.BoundClass(i) = k;
  if (k==3) && (LB(i)==UB(i))
    params.BoundClass(i) = 4;
  end
end

% transform starting values into their unconstrained
% surrogates. Check for infeasible starting guesses.
x0u = x0;
k=1;
for i = 1:n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      if x0(i)<=LB(i)
        % infeasible starting value. Use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(x0(i) - LB(i));
      end
      
      % increment k
      k=k+1;
    case 2
      % upper bound only
      if x0(i)>=UB(i)
        % infeasible starting value. use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(UB(i) - x0(i));
      end
      
      % increment k
      k=k+1;
    case 3
      % lower and upper bounds
      if x0(i)<=LB(i)
        % infeasible starting value
        x0u(k) = -pi/2;
      elseif x0(i)>=UB(i)
        % infeasible starting value
        x0u(k) = pi/2;
      else
        x0u(k) = 2*(x0(i) - LB(i))/(UB(i)-LB(i)) - 1;
        % shift by 2*pi to avoid problems at zero in fminsearch
        % otherwise, the initial simplex is vanishingly small
        x0u(k) = 2*pi+asin(max(-1,min(1,x0u(k))));
      end
      
      % increment k
      k=k+1;
    case 0
      % unconstrained variable. x0u(i) is set.
      x0u(k) = x0(i);
      
      % increment k
      k=k+1;
    case 4
      % fixed variable. drop it before fminsearch sees it.
      % k is not incremented for this variable.
  end
  
end
% if any of the unknowns were fixed, then we need to shorten
% x0u now.
if k<=n
  x0u(k:n) = [];
end

% were all the variables fixed?
if isempty(x0u)
  % All variables were fixed. quit immediately, setting the
  % appropriate parameters, then return.
  
  % undo the variable transformations into the original space
  x = xtransform(x0u,params);
  
  % final reshape
  x = reshape(x,xsize);
  
  % stuff fval with the final value
  fval = feval(params.fun,x,params.args{:});
  
  % fminsearchbnd was not called
  exitflag = 0;
  
  output.iterations = 0;
  output.funcCount = 1;
  output.algorithm = 'fminsearch';
  output.message = 'All variables were held fixed by the applied bounds';
  
  % return with no call at all to fminsearch
  return
end

% Check for an outputfcn. If there is any, then substitute my
% own wrapper function.
if ~isempty(options.OutputFcn)
  params.OutputFcn = options.OutputFcn;
  options.OutputFcn = @outfun_wrapper;
end

% now we can call fminsearch, but with our own
% intra-objective function.
[xu,fval,exitflag,output] = fminsearch(@intrafun,x0u,options,params);

% undo the variable transformations into the original space
x = xtransform(xu,params);

% final reshape to make sure the result has the proper shape
x = reshape(x,xsize);

% Use a nested function as the OutputFcn wrapper
  function stop = outfun_wrapper(x,varargin);
    % we need to transform x first
    xtrans = xtransform(x,params);
    
    % then call the user supplied OutputFcn
    stop = params.OutputFcn(xtrans,varargin{1:(end-1)});
    
  end

end % mainline end

% ======================================
% ========= begin subfunctions =========
% ======================================
function fval = intrafun(x,params)
% transform variables, then call original function

% transform
xtrans = xtransform(x,params);

% and call fun
fval = feval(params.fun,reshape(xtrans,params.xsize),params.args{:});

end % sub function intrafun end

% ======================================
function xtrans = xtransform(x,params)
% converts unconstrained variables into their original domains

xtrans = zeros(params.xsize);
% k allows some variables to be fixed, thus dropped from the
% optimization.
k=1;
for i = 1:params.n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      xtrans(i) = params.LB(i) + x(k).^2;
      
      k=k+1;
    case 2
      % upper bound only
      xtrans(i) = params.UB(i) - x(k).^2;
      
      k=k+1;
    case 3
      % lower and upper bounds
      xtrans(i) = (sin(x(k))+1)/2;
      xtrans(i) = xtrans(i)*(params.UB(i) - params.LB(i)) + params.LB(i);
      % just in case of any floating point problems
      xtrans(i) = max(params.LB(i),min(params.UB(i),xtrans(i)));
      
      k=k+1;
    case 4
      % fixed variable, bounds are equal, set it at either bound
      xtrans(i) = params.LB(i);
    case 0
      % unconstrained variable.
      xtrans(i) = x(k);
      
      k=k+1;
  end
end

end % sub function xtransform end





