function plotObjectStraight( rhoPhased, isoval, varargin )

    if nargin == 2
        skew = eye( 3 );
    else
        skew = varargin{1};
    end
    
%     meanang = sum( angle( rhoPhased(:) ).* abs( rhoPhased(:) ) ) / sum( abs( rhoPhased(:) ) );
    if nargin == 4
        rhoPhased = centerPhase( abs( rhoPhased ) .* exp( 1i * ( angle( rhoPhased ) ) ), varargin{2} );
    else
        rhoPhased = centerPhase( abs( rhoPhased ) .* exp( 1i * ( angle( rhoPhased ) ) ) );
    end

    [ x, y, z ] = meshgrid( 1:size( rhoPhased, 1 ), 1:size( rhoPhased, 2 ), 1:size( rhoPhased, 3 ) );
    y = max( y(:) ) - y;
    
    x = x - mean( x(:) );
    y = y - mean( y(:) );
    z = z - mean( z(:) );
    
    pts = skew * [ x(:) y(:) z(:) ]';
    
    x = reshape( pts(1,:)', size( rhoPhased ) );
    y = reshape( pts(2,:)', size( rhoPhased ) );
    z = reshape( pts(3,:)', size( rhoPhased ) );
    
    
    
%     fig = figure;
    iso = isosurface( ...
        x, y, z, ...
        smooth3( smooth3( abs( rhoPhased ), 'gaussian', 13 ), 'gaussian', 13 ), ...
        isoval, ...
        smooth3( smooth3( angle( rhoPhased ), 'gaussian', 13 ), 'gaussian', 13 ) ...
    );


    iso.vertices = iso.vertices - repmat( mean( iso.vertices, 1 ), size( iso.vertices, 1 ), 1 );
%     size( iso.vertices )
%     iso.vertices = ( skew * ( iso.vertices )' )';

    patch( iso, 'FaceColor', 'interp', 'EdgeColor', 'none' );
%     axis image;
%     axis off;
    grid on;
    colormap( 'parula' );
    colorbar;
    camlight( 0, 0 );
    camlight( 120, 0 );
    camlight( 240, 0 );
    camlight( 0, 90 );
    camlight( 0, -90 );

end