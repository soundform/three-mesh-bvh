export const bvh_raymarch_functions = /* glsl */`

float _distToPoints(
	// geometry info and points range
	sampler2D positionAttr, usampler2D indexAttr, uint offset, uint count,

	// point and cut off range
	vec3 point, float closestDistanceSquared
) {

	for ( uint i = offset; i < offset + count; i ++ ) {
    vec3 localBarycoord = vec3(1,0,0);
    uvec3 indices = uTexelFetch1D( indexAttr, i ).xyz;
		vec3 closestPoint = texelFetch1D( positionAttr, indices.x ).xyz;
		vec3 delta = point - closestPoint;
		float sqDist = dot( delta, delta );

    if ( sqDist < closestDistanceSquared )
			closestDistanceSquared = sqDist;
	}

	return closestDistanceSquared;

}

float _distToNode( vec3 point, sampler2D bvhBounds, uint currNodeIndex ) {

	vec3 boundsMin = texelFetch1D( bvhBounds, currNodeIndex * 2u + 0u ).xyz;
	vec3 boundsMax = texelFetch1D( bvhBounds, currNodeIndex * 2u + 1u ).xyz;
	vec3 delta = point - clamp( point, boundsMin, boundsMax );
	return dot( delta, delta );

}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhClosestPointToPoint(\
		bvh,\
		point, maxDistance)\
	_bvhClosestPointToPoint(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		point, maxDistance)

float _bvhClosestPointToPoint(
	// bvh info
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,

	// point to check
	vec3 point, float maxDistance
 ) {

	// stack needs to be twice as long as the deepest tree we expect because
	// we push both the left and right child onto the stack every traversal
	int ptr = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

	float closestDistanceSquared = maxDistance * maxDistance;
	bool found = false;
	while ( ptr >= 0 && ptr < BVH_STACK_DEPTH ) {

		uint currNodeIndex = stack[ ptr-- ];

		// check if we intersect the current bounds
		float boundsHitDistance = _distToNode( point, bvh_bvhBounds, currNodeIndex );
		if ( boundsHitDistance > closestDistanceSquared ) {

			continue;

		}

		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, currNodeIndex ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;
			closestDistanceSquared = _distToPoints(
				bvh_position, bvh_index, offset, count, point, closestDistanceSquared);

		} else {

			uint leftIndex = currNodeIndex + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = boundsInfo.y;
      float lhs = _distToNode( point, bvh_bvhBounds, leftIndex );
      float rhs = _distToNode( point, bvh_bvhBounds, rightIndex );
			bool leftToRight = lhs < rhs; // rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;

			// set c2 in the stack so we traverse it later. We need to keep track of a pointer in
			// the stack while we traverse. The second pointer added is the one that will be
			// traversed first
			stack[ ++ptr ] = c2;
			stack[ ++ptr ] = c1;

		}

	}

	return sqrt( closestDistanceSquared );

}
`;
