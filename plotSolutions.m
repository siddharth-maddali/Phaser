load( 'S0032_phased-3.mat' );

% Manually removing phase ramp from object
frho = fftshift( fftn( fftshift( rhoPhased ) ) );
sz = size( frho );
[ ~, idx ] = max( abs( frho(:) ).^2 );
[ i, j, k ] = ind2sub( size( frho ), idx );
frho = circshift( frho, [ sz(1)/2-i, sz(2)/2-j, sz(3)/2-k ] );
rhoPhased = fftshift( ifftn( fftshift( frho ) ) );
img = rhoPhased;

% plotting
figure; 
s1 = subplot( 1, 2, 1 ); 
plotObjectStraight( img, 4e-3, Breal ); 
colorbar; 
set( gca, 'FontSize', 20 ); 
axis on; 
grid on; 
view( 3 ); 
axis image;
alpha( 0.75 ); 
hold on;
q = quiver3( [ 0 0 0 ], [ 0 0 0 ], [ 0 0 0 ], Breal(1,:), Breal(2,:), Breal(3,:), 10, 'r', 'LineWidth', 2 );
title( '$\rho (\mathbf{x})$', 'interpreter', 'latex' );

s2 = subplot( 1, 2, 2 ); 
plotObjectTwin( img, 4e-3, Breal ); 
colorbar; 
set( gca, 'FontSize', 20 ); 
axis on; 
grid on; 
view( 3 ); 
axis image;
alpha( 0.75 ); 
hold on;
q = quiver3( [ 0 0 0 ], [ 0 0 0 ], [ 0 0 0 ], Breal(1,:), Breal(2,:), Breal(3,:), 10, 'r', 'LineWidth', 2 );
title( '$\rho^*(-\mathbf{x})$', 'interpreter', 'latex' );

linkprop( [ s1 s2 ], { ...
'CameraPosition', ...
'CameraTarget', ...
'CameraUpVector', ...
'CameraViewAngle' } );
