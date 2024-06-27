function seg = slic3Dhsi(im, k, m, seRadius, nItr)

%% Initialization
    if ~exist('seRadius','var')   || isempty(seRadius),     seRadius = 1;     end
    if ~exist('nItr','var')   || isempty(nItr),     nItr = 10;     end
    [rows, cols, L] = size(im);
   
    S = sqrt(rows*cols / (k * sqrt(3)/2));
    countCols = round(cols/S - 0.5);
    S = cols/(countCols + 0.5); 
    countRows = round(rows/(sqrt(3)/2*S));
    vSpacing = rows/countRows;

    % Recompute the number of superpixels k
    k = countRows * countCols;
    
    % Allocate memory and initialise clusters, labels and distances.
    C = zeros(L+3,k);        % Cluster centre data  1:L is mean spectral value,
                             % L+1: L+2 is row, col of centre, 
                             % L+3 is the number of pixels in each clustering
    l = -ones(rows, cols);   % Pixel labels.
    d = inf(rows, cols);     % Pixel distances from cluster centres.
    
    % Initialise clusters on a hexagonal grid
    kk = 1;
    r = vSpacing/2;
    for ri = 1:countRows
        if mod(ri,2), c = S/2; else, c = S;  end
        for ci = 1:countCols
            cc = round(c); rr = round(r);
            C(1:L+2, kk) = [squeeze(im(rr,cc,:)); cc; rr];
            c = c+S;
            kk = kk+1;
        end
        r = r+vSpacing;
    end
    S = round(S);
    
 %  Upadate cluster centers   
    for n = 1:nItr
       for kk = 1:k
           % Get subimage around cluster
           rmin = max(C(L+2,kk)-S, 1);   rmax = min(C(L+2,kk)+S, rows); 
           cmin = max(C(L+1,kk)-S, 1);   cmax = min(C(L+1,kk)+S, cols); 
           subim = im(rmin:rmax, cmin:cmax, :);  
           assert(numel(subim) > 0);
           % Calculate distance
           D = dist(C(:, kk), subim, rmin, cmin, S, m);
           % If any pixel distance from the cluster centre is less than its
           % previous value update its distance and label
           subd =  d(rmin:rmax, cmin:cmax);
           subl =  l(rmin:rmax, cmin:cmax);
           updateMask = D < subd;
           subd(updateMask) = D(updateMask);
           subl(updateMask) = kk;
           d(rmin:rmax, cmin:cmax) = subd;
           l(rmin:rmax, cmin:cmax) = subl;           
       end
       % Update cluster centres with mean spectral values
       C(:) = 0;
       for r = 1:rows
           for c = 1:cols
              spectra = reshape(im(r,c,:),L,1);
              tmp = [spectra; c; r; 1];
              C(:, l(r,c)) = C(:, l(r,c)) + tmp;
           end
       end
       % Divide by number of pixels in each superpixel to get mean spectrum
       for kk = 1:k 
           C(1:L,kk) = C(1:L,kk)/C(L+3,kk); 
           C(L+1:L+2,kk) = round(C(L+1:L+2,kk)/C(L+3,kk)); 
       end
    end
%% Cleanup small orphaned regions.     
    % The cleaned up regions are assigned to the nearest cluster.
    if seRadius
        [l, Am] = mcleanupregions(l, seRadius);
    else
        l = makeregionsdistinct(l);
        l = renumberregions(l);
        Am = regionadjacency(l);    
    end
    
%% recalculate the center
    N = length(Am);
    C = zeros(L+3,N); 
    for r = 1:rows
       for c = 1:cols
          spec = reshape(im(r,c,:),L,1);
          tmp = [spec; c; r; 1];
          C(:, l(r,c)) = C(:, l(r,c)) + tmp;
       end
    end
    % Divide by number of pixels in each superpixel to get mean values
    for kk = 1:N
       C(1:L,kk) = C(1:L,kk)/C(L+3,kk); 
       C(L+1:L+2,kk) = round(C(L+1:L+2,kk)/C(L+3,kk)); 
    end
    % Calculate the confidence index
    d_c = zeros(rows,cols); % Spectral distance
    d_s = zeros(rows,cols); % Spatial distance
    for r = 1:rows
        for c = 1:cols
            lab = l(r,c); 
            Cspec = C(1:L,lab);
            spec= reshape(im(r,c,:),L,1);
            d_c(r,c) = acos(min(max(dot(spec,Cspec)/(norm(spec)*norm(Cspec)),0.00000),1.000000));%
            d_s(r,c) = (r - C(L+2,lab)).^2+(c - C(L+1,lab)).^2;
        end
    end
    d_c=d_c./max(d_c(:));
    Cj = 1./(sqrt(m*d_c + (1-m)*d_s/S.^2)+eps);
    
   % Segmentation results  
    seg.X_c = C(1:L,:);   % the averange spectra of superpixels
    seg.P = N;  % the number of superpixels
    seg.Cj = reshape(Cj,rows*cols ,1);  % the confidence index
    seg.labels = reshape(l,rows*cols ,1); 
    seg.Sw = S; 
    
   % Plot results

    figure;
    im_seg = showsegresults(im,l);
    subplot_tight(1, 2, 1,[.01 .01]); image(im_seg);axis image;axis off;title('Superpixels','fontsize',8);  
    subplot_tight(1, 2, 2,[.01 .01]); imagesc(Cj);axis off;axis image;title('Confindence index','fontsize',8);   
    drawnow;

    %saveas(gcf,'../results/results_Seg.tif');
end
       
    
%%%%%%%%%%%%%%%%%%%%%%  dist  %%%%%%%%%%%%%%%%%%%%%%
%
% Usage:  D = dist(C, im, r1, c1, S, m)
% 
% Arguments:   C - Cluster center
%             im - sub-image surrounding cluster center
%         r1, c1 - row and column of top left corner of sub image within the
%                  overall image.
%              S - grid spacing
%              m - weighting factor between spectral and spatial distances.
%
% Returns:     D - Distance image giving distance of every pixel in the
%                  subimage from the cluster center
%
% Distance = sqrt( dc^2 + (ds/S)^2*m^2 )
% where:
% dc = arccos(ims*imc/sqrt(sum(ims.^2,2)*sum(imc.^2)))  % Spectral distance
% ds = sqrt(dx^2 + dy^2)                                % Spatial distance

function D = dist(C, im, r1, c1, S, m) 
    [rows, cols, L] = size(im);  
    % Squared colour difference
    ims = reshape(im,rows*cols,L);imc = C(1:L); 
    dcos = (ims * imc)./(sqrt(sum(ims.^2,2)) * norm(imc));
    dc2 = acos(min(max(reshape(dcos,rows,cols),0.0000),1.0000));
    dc2 = dc2./max(dc2(:));
    % Squared spatial distance
    [x,y] = meshgrid(c1:(c1+cols-1),r1:(r1+rows-1));
    ds2 = (x-C(L+1)).^2 + (y-C(L+2)).^2;
    D = sqrt(m*dc2 + (1-m) * ds2 ./ S^2 );
end

% Show segmentation results
% Usage: Img_seg = showsegresults(Y,labels);
%==========================================================================
function Img_seg = showsegresults(Y,labels)
[p,q,L] = size(Y);
Img=zeros(p,q,3); lim = [round(L/4),round(L/2),round(3*L/4)];
for itr= 1:3
    Sp = Y(:,:,lim(itr));
    X_r = sort(reshape(Sp,p*q,1)); 
    a=X_r(round(0.02*p*q));
    b=X_r(round(0.98*p*q));
    for i=1:p
        for j=1:q
            if Sp(i,j) < a
                Img(i,j,itr) = a;
            else
                if Sp(i,j)< b
                Img(i,j,itr) = Sp(i,j);
                else
                Img(i,j,itr) = b;
                end
            end
        end    
    end
    Img(:,:,itr)  =(Sp -a).*(255/(b-a));
end
Img=uint8(Img);
Img_seg = drawregionboundaries(labels, Img, [255 0 0]);
end

% MCLEANUPREGIONS  Morphological clean up of small segments in an image of segmented regions
%
% Usage: [seg, Am] = mcleanupregions(seg, seRadius)
%
% Arguments: seg - A region segmented image, such as might be produced by a
%                  graph cut algorithm.  All pixels in each region are labeled
%                  by an integer.
%       seRadius - Structuring element radius.  This can be set to 0 in which
%                  case  the function will simply ensure all labeled regions
%                  are distinct and relabel them if necessary. 
%
% Returns:   seg - The updated segment image.
%             Am - Adjacency matrix of segments.  Am(i, j) indicates whether
%                  segments labeled i and j are connected/adjacent
%
% Typical application:
% If a graph cut or superpixel algorithm fails to converge stray segments
% can be left in the result.  This function tries to clean things up by:
% 1) Checking there is only one region for each segment label. If there is
%    more than one region they are given unique labels.
% 2) Eliminating regions below the structuring element size
%
% Note that regions labeled 0 are treated as a 'privileged' background region
% and is not processed/affected by the function.
%
% See also: REGIONADJACENCY, RENUMBERREGIONS, CLEANUPREGIONS, MAKEREGIONSDISTINCT

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% March   2013 
% June    2013  Improved morphological cleanup process using distance map

function [seg, Am, mask] = mcleanupregions(seg, seRadius)
option = 2;
    % 1) Ensure every segment is distinct 
    [seg, maxlabel] = makeregionsdistinct(seg);
    
    % 2) Perform a morphological opening on each segment, subtract the opening
    % from the orignal segment to obtain regions to be reassigned to
    % neighbouring segments.
    if seRadius
        se = circularstruct(seRadius);   % Accurate and not noticeably slower
                                         % if radius is small
%       se = strel('disk', seRadius, 4);  % Use approximated disk for speed
        mask = zeros(size(seg));

        if option == 1        
            for l = 1:maxlabel
                b = seg == l;
                mask = mask | (b - imopen(b,se));
            end
            
        else   % Rather than perform a morphological opening on every
               % individual region in sequence the following finds separate
               % lists of unconnected regions and performs openings on these.
               % Typically an image can be covered with only 5 or 6 lists of
               % unconnected regions.  Seems to be about 2X speed of option
               % 1. (I was hoping for more...)
            list = finddisconnected(seg);
            
            for n = 1:length(list)
                b = zeros(size(seg));
                for m = 1:length(list{n})
                    b = b | seg == list{n}(m);
                end

                mask = mask | (b - imopen(b,se));
            end
        end
        
        % Compute distance map on inverse of mask
        [~, idx] = bwdist(~mask);
        
        % Assign a label to every pixel in the masked area using the label of
        % the closest pixel not in the mask as computed by bwdist
        seg(mask) = seg(idx(mask));
    end
    
    % 3) As some regions will have been relabled, possibly broken into several
    % parts, or absorbed into others and no longer exist we ensure all regions
    % are distinct again, and renumber the regions so that they sequentially
    % increase from 1.  We also need to reconstruct the adjacency matrix to
    % reflect the changed number of regions and their relabeling.

    seg = makeregionsdistinct(seg);
    [seg, minLabel, maxLabel] = renumberregions(seg);
    Am = regionadjacency(seg);    
    
end

% MAKEREGIONSDISTINCT Ensures labeled segments are distinct
%
% Usage: [seg, maxlabel] = makeregionsdistinct(seg, connectivity)
%
% Arguments: seg - A region segmented image, such as might be produced by a
%                  superpixel or graph cut algorithm.  All pixels in each
%                  region are labeled by an integer.
%   connectivity - Optional parameter indicating whether 4 or 8 connectedness
%                  should be used.  Defaults to 4.
%
% Returns:   seg - A labeled image where all segments are distinct.
%       maxlabel - Maximum segment label number.
%
% Typical application: A graphcut or superpixel algorithm may terminate in a few
% cases with multiple regions that have the same label.  This function
% identifies these regions and assigns a unique label to them.
%
% See also: SLIC, CLEANUPREGIONS, RENUMBERREGIONS

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% June 2013


function [seg, maxlabel] = makeregionsdistinct(seg, connectivity)
    
    if ~exist('connectivity', 'var'), connectivity = 8; end
    
    % Ensure every segment is distinct but do not touch segments 
    % with a label of 0
    labels = unique(seg(:))';
    maxlabel = max(labels);
    labels = setdiff(labels,0);  % Remove 0 from the label list
    
    for l = labels
        [bl,num] = bwlabel(seg==l, connectivity);  
        
        if num > 1  % We have more than one region with the same label
            for n = 2:num
                maxlabel = maxlabel+1;  % Generate a new label
                seg(bl==n) = maxlabel;  % and assign to this segment
            end
        end
    end

end

% CIRCULARSTRUCT
%
% Function to construct a circular structuring element
% for morphological operations.
%
% function strel = circularstruct(radius)
%
% Note radius can be a floating point value though the resulting
% circle will be a discrete approximation
%
% Peter Kovesi   March 2000

function strel = circularstruct(radius)

if radius < 1
  error('radius must be >= 1');
end

dia = ceil(2*radius);  % Diameter of structuring element

if mod(dia,2) == 0     % If diameter is a odd value
 dia = dia + 1;        % add 1 to generate a `centre pixel'
end

r = fix(dia/2);
[x,y] = meshgrid(-r:r);
rad = sqrt(x.^2 + y.^2);  
strel = rad <= radius;

end

% FINDDISCONNECTED find groupings of disconnected labeled regions
%
% Usage: list = finddisconnected(l)
%
% Argument:   l - A labeled image segmenting an image into regions, such as
%                 might be produced by a graph cut or superpixel algorithm.
%                 All pixels in each region are labeled by an integer.
%
% Returns: list - A cell array of lists of regions that are not
%                 connected. Typically there are 5 to 6 lists.
%
% Used by MCLEANUPREGIONS to reduce the number of morphological closing
% operations 
%
% See also: MCLEANUPREGIONS, REGIONADJACENCY

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% PK July 2013


function list = finddisconnected(l)
 
    debug = 0;
    [Am, Al] = regionadjacency(l);
    
    N = max(l(:));  % number of labels
    
    % Array for keeping track of visited labels
    visited = zeros(N,1);

    list = {};
    listNo = 0;
    for n = 1:N
        if ~visited(n)
            listNo = listNo + 1;
            list{listNo} = n;
            visited(n) = 1;
            
            % Find all regions not directly connected to n and not visited
            notConnected = setdiff(find(~Am(n,:)), find(visited));
            
            % For each unconnected region check that it is not already
            % connected to a region in the list. If not, add to list
            for m = notConnected
                if isempty(intersect(Al{m}, list{listNo}))
                    list{listNo} = [list{listNo} m];
                    visited(m) = 1;
                end
            end
         end % if not visited(n)
        
    end
    
    % Display each list of unconncted regions as an image
    if debug   
        for n = 1:length(list)
            
            mask = zeros(size(l));
            for m = 1:length(list{n})
                mask = mask | l == list{n}(m);
            end
            
            fprintf('list %d of %d length %d \n', n, length(list), length(list{n}))
            show(mask);
            keypause
        end
    end
end

% REGIONADJACENCY Computes adjacency matrix for image of labeled segmented regions
%
% Usage:  [Am, Al] = regionadjacency(L, connectivity)
%
% Arguments:  L - A region segmented image, such as might be produced by a
%                 graph cut or superpixel algorithm.  All pixels in each
%                 region are labeled by an integer.
%  connectivity - 8 or 4.  If not specified connectivity defaults to 8.
%
% Returns:   Am - An adjacency matrix indicating which labeled regions are
%                 adjacent to each other, that is, they share boundaries. Am
%                 is sparse to save memory.
%            Al - A cell array representing the adjacency list corresponding
%                 to Am.  Al{n} is an array of the region indices adjacent to
%                 region n.
%
% Regions with a label of 0 are not processed. They are considered to be
% 'background regions' that are not to be considered.  If you want to include
% these regions you should assign a new positive label to these areas using, say
% >> L(L==0) = max(L(:)) + 1;
%
% See also: CLEANUPREGIONS, RENUMBERREGIONS, SLIC

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% February 2013  Original version
% July     2013  Speed improvement in sparse matrix formation (4x)

function  [Am, varargout] = regionadjacency(L, connectivity)

    if ~exist('connectivity', 'var'), connectivity = 8; end
    [rows,cols] = size(L);
    
    % Identify the unique labels in the image, excluding 0 as a label.
    labels = setdiff(unique(L(:))',0);

    if isempty(labels)
        warning('There are no objects in the image')
        Am = [];
        Al = {};
        return
    end

    N = max(labels);    % Required size of adjacency matrix
    
    % Strategy:  Step through the labeled image.  For 8-connectedness inspect 
    % pixels as follows and set the appropriate entries in the adjacency
    % matrix. 
    %      x - o
    %    / | \
    %  o   o   o
    %
    % For 4-connectedness we only inspect the following pixels
    %      x - o
    %      | 
    %      o  
    %
    % Becuase the adjacency search looks 'forwards' a final OR operation is
    % performed on the adjacency matrix and its transpose to ensure
    % connectivity both ways.

    % Allocate vectors for forming row, col, value triplets used to construct
    % sparse matrix.  Forming these vectors first is faster than filling
    % entries directly into the sparse matrix
    i = zeros(rows*cols,1);  % row value
    j = zeros(rows*cols,1);  % col value
    s = zeros(rows*cols,1);  % value
    
    if connectivity == 8
        n = 1;
        for r = 1:rows-1

            % Handle pixels in 1st column
            i(n) = L(r,1); j(n) = L(r  ,2); s(n) = 1; n=n+1;
            i(n) = L(r,1); j(n) = L(r+1,1); s(n) = 1; n=n+1;
            i(n) = L(r,1); j(n) = L(r+1,2); s(n) = 1; n=n+1;
            
            % ... now the rest of the column
            for c = 2:cols-1
               i(n) = L(r,c); j(n) = L(r  ,c+1); s(n) = 1; n=n+1;
               i(n) = L(r,c); j(n) = L(r+1,c-1); s(n) = 1; n=n+1;
               i(n) = L(r,c); j(n) = L(r+1,c  ); s(n) = 1; n=n+1;
               i(n) = L(r,c); j(n) = L(r+1,c+1); s(n) = 1; n=n+1;
            end
        end
        
    elseif connectivity == 4
        n = 1;
        for r = 1:rows-1
            for c = 1:cols-1
                i(n) = L(r,c); j(n) = L(r  ,c+1); s(n) = 1; n=n+1;
                i(n) = L(r,c); j(n) = L(r+1,c  ); s(n) = 1; n=n+1;
            end
        end
    
    else
        error('Connectivity must be 4 or 8');
    end
    
    % Form the logical sparse adjacency matrix
    Am = logical(sparse(i, j, s, N, N)); 
    
    % Zero out the diagonal 
    for r = 1:N
        Am(r,r) = 0;
    end
    
    % Ensure connectivity both ways for all regions.
    Am = Am | Am';
    
    % If an adjacency list is requested...
    if nargout == 2
        Al = cell(N,1);
        for r = 1:N
            Al{r} = find(Am(r,:));
        end
        varargout{1} = Al;
    end
    

end

% RENUMBERREGIONS
%
% Usage: [nL, minLabel, maxLabel] = renumberregions(L)
%
% Argument:   L - A labeled image segmenting an image into regions, such as
%                 might be produced by a graph cut or superpixel algorithm.
%                 All pixels in each region are labeled by an integer.
%
% Returns:   nL - A relabeled version of L so that label numbers form a
%                 sequence 1:maxRegions  or 0:maxRegions-1 depending on
%                 whether L has a region labeled with 0s or not.
%      minLabel - Minimum label in the renumbered image.  This will be 0 or 1.
%      maxLabel - Maximum label in the renumbered image.
%
% Application: Segmentation algorithms can produce a labeled image with a non
% contiguous numbering of regions 1 4 6 etc. This function renumbers them into a
% contiguous sequence.  If the input image has a region labeled with 0s this
% region is treated as a privileged 'background region' and retains its 0
% labeling. The resulting image will have labels ranging over 0:maxRegions-1.
% Otherwise the image will be relabeled over the sequence 1:maxRegions
%
% See also: CLEANUPREGIONS, REGIONADJACENCY

% Copyright (c) 2010 Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% October  2010
% February 2013 Return label numbering range

function [nL, minLabel, maxLabel] = renumberregions(L)

    nL = L;
    labels = unique(L(:))';  % Sorted list of unique labels
    N = length(labels);
    
    % If there is a label of 0 we ensure that we do not renumber that region
    % by removing it from the list of labels to be renumbered.
    if labels(1) == 0
        labels = labels(2:end);
        minLabel = 0;
        maxLabel = N-1;
    else
        minLabel = 1;
        maxLabel = N;
    end
    
    % Now do the relabelling
    count = 1;
    for n = labels
        nL(L==n) = count;
        count = count+1;
    end
    
end

% DRAWREGIONBOUNDARIES Draw boundaries of labeled regions in an image
%
% Usage: maskim = drawregionboundaries(l, im, col)
%
% Arguments:
%            l - Labeled image of regions.
%           im - Optional image to overlay the region boundaries on.
%          col - Optional colour specification. Defaults to black.  Note that
%                the colour components are specified as values 0-255.
%                For example red is [255 0 0] and white is [255 255 255].
%
% Returns: 
%       maskim - If no image has been supplied maskim is a binary mask
%                image indicating where the region boundaries are.
%                If an image has been supplied maskim is the image with the
%                region boundaries overlaid 
%
% See also: MASKIMAGE

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% Feb 2013

function maskim = drawregionboundaries(l, im, col)
    
    % Form the mask by applying a sobel edge detector to the labeled image,
    % thresholding and then thinning the result.
%    h = [1  0 -1
%         2  0 -2
%         1  0 -1];
    h = [-1 1];  % A simple small filter is better in this application.
                 % Small regions 1 pixel wide get missed using a Sobel
                 % operator 
    gx = filter2(h ,l);
    gy = filter2(h',l);
    maskim = (gx.^2 + gy.^2) > 0;
    maskim = bwmorph(maskim, 'thin', Inf);
    
    % Zero out any mask values that may have been set around the edge of the
    % image.
    maskim(1,:) = 0; maskim(end,:) = 0;
    maskim(:,1) = 0; maskim(:,end) = 0;
    
    % If an image has been supplied apply the mask to the image and return it 
    if exist('im', 'var') 
        if ~exist('col', 'var'), col = 0; end
        maskim = maskimage(im, maskim, col);
    end
end

% MASKIMAGE Apply mask to image
%
% Usage: maskedim = maskimage(im, mask, col)
%
% Arguments:    im  - Image to be masked
%             mask  - Binary masking image
%              col  - Value/colour to be applied to regions where mask == 1
%                     If im is a colour image col can be a 3-vector
%                     specifying the colour values to be applied.
%
% Returns: maskedim - The masked image
%
% See also; DRAWREGIONBOUNDARIES

% Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
%
% Feb 2013

function maskedim = maskimage(im, mask, col)
    
    [~,~, chan] = size(im);
    
    % Set default colour to 0 (black)
    if ~exist('col', 'var'), col = 0; end
    
    % Ensure col has same length as image depth.
    if length(col) == 1
        col = repmat(col, [chan 1]);
    else
        assert(length(col) == chan);
    end
    
    % Perform masking
    maskedim = im;
    for n = 1:chan
        tmp = maskedim(:,:,n);
        tmp(mask) = col(n);
        maskedim(:,:,n) = tmp;
    end
end
