function [ iso, x, y, z, p, cbar ] = plotParticle( rho, varargin )

%     varargins:
%     1 = isoval (float) (default = 0.5)
%     2 = skew (3x3 matrix) (real space sampling basis)
%     3 = objSelect (string) ('straight': plots rho(x), 'twin': plots rho*(-x) )
%     4 = smoothflag (string) ('smooth' or 'nosmooth' for phase)

    % first resolving all inputs
    isoval = 0.5;               %
    skew = eye( 3 );            %   defaults
    objSelect = 'straight';     %
    smth = 'smooth';            %

    if nargin > 1
        isoval = varargin{1};
    end
    if nargin > 2
        skew = varargin{2};
    end
    if nargin > 3
        objSelect = varargin{3};
    end
    if nargin > 4
        smth = varargin{4};
    end
    if nargin > 5
        fprintf( 2, 'plotParticle warning: Too many input arguments. \n' );
    end
    
% %     setting average phase to zero
%     normalizr = sum( abs( rho(:) ) );
%     phimean = sum( angle( rho(:) ) .* abs( rho(:) ) ) / normalizr;
    
    if strcmp( objSelect, 'straight' )
        rhoFinal = rho; 
        % i.e. the solution rho(x)
    elseif strcmp( objSelect, 'twin' )
        rhoFinal = conj( flip( flip( flip( rho, 1 ), 2 ), 3 ) );
        % i.e. the solution rho*(-x)
    else
        fprintf( 2, 'plotParticle warning: Unrecognized object convention. \n' );
        rhoFinal = rho;
    end
    amp = abs( rhoFinal );
    ang = angle( rhoFinal );
%     ang = ang .* ( ang > 0 ) + ( 2*pi + ang ) .* ( ang < 0 ); % unwrapping
%     angmean = sum( ang(:) .* amp(:)  ) / sum( amp(:) );
%     ang = ang - angmean;
    rhoPlot = amp .* exp( 1j * ang );
    
    [ x, y, z ] = meshgrid( 1:size( rhoPlot, 1 ), 1:size( rhoPlot, 2 ), 1:size( rhoPlot, 3 ) );
    y = max( y(:) ) - y;
    
    x = x - mean( x(:) );
    y = y - mean( y(:) );
    z = z - mean( z(:) );
    
    pts = skew * [ x(:) y(:) z(:) ]';        
    x = reshape( pts(1,:)', size( rhoPlot ) );
    y = reshape( pts(2,:)', size( rhoPlot ) );
    z = reshape( pts(3,:)', size( rhoPlot ) );
    if strcmp( smth, 'smooth' )
        colorField = smooth3( angle( rhoPlot ), 'gaussian', 1 );
    else
        colorField = angle( rhoPlot );
    end
    
    iso = isosurface( ...
        x, y, z, ...
        smooth3( smooth3( abs( rhoPlot ), 'gaussian', 3 ), 'gaussian', 3 ), ...
        isoval, ...
        colorField ...
    );
    iso.facevertexcdata = iso.facevertexcdata - mean( iso.facevertexcdata(:) );
    p = patch( iso, 'FaceColor', 'interp', 'EdgeColor', 'none' );
    axis image;
    grid on;
    colormap( 'parula' );
    cbar = colorbar;
    tetLighting();
end