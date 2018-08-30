function [ imgC ] = centerPhase( img, varargin )
	shp = size( img );
	imgang = angle( img );
	if nargin == 1
		shft = imgang( 1+shp(1)/2, 1+shp(2)/2, 1+shp(3)/2 );
	else
		shft = varargin{1};
	end
	
	imgang = imgang - shft;
	imgC = abs( img ) .* exp( 1i * imgang );
end
