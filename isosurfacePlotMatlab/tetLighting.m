% Tetrahedral lighting: applies camera lighting as if the object is at the
% center of a regular tetrahedron and the light sources are at the
% vertices, looking inwards.

function tetLighting()
    verts = [ ...
        [  1,  0, -1/sqrt( 2 ) ]; ...
        [ -1,  0, -1/sqrt( 2 ) ]; ...
        [  0,  1,  1/sqrt( 2 ) ]; ...
        [  0, -1,  1/sqrt( 2 ) ]  ...
    ]';
    verts = verts - repmat( mean( verts, 2 ), 1, size( verts, 2 ) );
    verts = verts ./ repmat( sqrt( sum( verts.^2 ) ), 3, 1 );
    theta = 180 / pi * acos( verts(3,:) );
    phi = 180 / pi * atan2( verts(2,:), verts(1,:) );
    for n = 1:4
        camlight( theta( n ), phi( n ) );
    end
end