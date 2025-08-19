export const bvh_raymarching_functions = /* glsl */`

struct PointsRange {
  float dist;
  uint commonNode; 
  vec3 closestPoint; 
  int countPts, lookupsPts, lookupsBVH; 
};

float _distToBox(vec3 p, vec3 aa, vec3 bb) {
  return length(p - clamp(p, aa, bb));
}

float _distToNode( sampler2D bvhBounds, uint nodeId, vec3 point, vec3 margin ) {
	vec3 boundsMin = texelFetch1D( bvhBounds, nodeId * 2u + 0u ).xyz - margin;
	vec3 boundsMax = texelFetch1D( bvhBounds, nodeId * 2u + 1u ).xyz + margin;
  return _distToBox(point, boundsMin, boundsMax);
}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhClosestPointToPoint(bvh, pointSize, point, margin, maxDist, rootNode)\
	_bvhClosestPointToPoint(bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
    pointSize, point, margin, maxDist, rootNode)

PointsRange _bvhClosestPointToPoint(
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,
	float pointSize, vec3 point, vec3 margin, float maxDist, uint rootNode) {

	uint stack[ BVH_STACK_DEPTH ];
  int ptr = 0;
	stack[ 0 ] = rootNode;

  // chain[i] = parent of stack[i] and stack[i+1]
  uint chain [ BVH_STACK_DEPTH ];
  int fork = -1; // fork < ptr
  chain[ 0 ] = 0xffffffffu;

  PointsRange pr;
  vec3 pointAA = point - margin;
  vec3 pointBB = point + margin;

	while ( ptr >= 0 && ptr < BVH_STACK_DEPTH ) {

    if (ptr < fork)
      fork = ptr;

    pr.lookupsBVH++;

    uint nodeId = stack[ ptr-- ];
		float boxDist = _distToNode( bvh_bvhBounds, nodeId, point, margin + pointSize );
    if ( boxDist > maxDist )
      continue;

		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, nodeId ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;
			int numPts = 0;

      // check what points overlap with the (aa, bb) box
      for ( uint i = offset; i < offset + count; i++ ) {
        uint index = uTexelFetch1D( bvh_index, i ).x;
        vec3 p = texelFetch1D( bvh_position, index ).xyz;
        float dist = max(length(p - point) - pointSize, 0.);

        if (dist < maxDist) {
          maxDist = dist;
          pr.closestPoint = p;
        }

        // check if sphere (p, pointSize) intersects with box (aa, bb)
        if (p == clamp(p, pointAA - pointSize, pointBB + pointSize))
          numPts++;
      }

      pr.lookupsPts += int(count); // number of points fetched
      pr.countPts += numPts; // number of points matched

      // if there are matching points in this leaf node, update the common BVH node
      if (numPts > 0) {
        if (fork < 0) {
          fork = ptr; // it's the first leaf node found
          pr.commonNode = nodeId;
        } else {
          pr.commonNode = chain[fork]; // move up the tree
        }
      }

    } else {

      uint leftIndex = nodeId + 1u;
      uint rightIndex = boundsInfo.y;
      //uint splitAxis = boundsInfo.x & 0x0000ffffu;
      pr.lookupsBVH += 2;
      float lhs = _distToNode( bvh_bvhBounds, leftIndex, point, vec3(0) );
      float rhs = _distToNode( bvh_bvhBounds, rightIndex, point, vec3(0) );
      bool leftToRight = lhs < rhs; // rayDirection[ splitAxis ] >= 0.0;
      uint c1 = leftToRight ? leftIndex : rightIndex;
      uint c2 = leftToRight ? rightIndex : leftIndex;

      stack[ ++ptr ] = c2; // to be inspected last
      stack[ ++ptr ] = c1; // to be inspected first

      if (fork < 0)
        chain[ ptr - 1 ] = nodeId;
    }
	}

  pr.dist = maxDist;
	return pr;
}`;
