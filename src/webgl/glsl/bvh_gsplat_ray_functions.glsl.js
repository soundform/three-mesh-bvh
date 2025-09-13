export const bvh_gsplat_ray_functions = /* glsl */`

#ifndef MAX_SPLATS_PER_RAY
#define MAX_SPLATS_PER_RAY 1
#endif

struct BVHIntersectResult {
  int count; // 0..MAX_SPLATS_PER_RAY
  float[MAX_SPLATS_PER_RAY] dist; // 0 <= dist[i] <= dist[i+1]; INFINITY if no intersection
  uint[MAX_SPLATS_PER_RAY] splatId;

  // stats
  uint numLookupsBVH;
  uint numLookupsSplats;
};

vec2 rayBox(vec3 ro, vec3 rd, vec3 aa, vec3 bb) {
    vec3 ird = 1./rd;
    vec3 tbot = ird*(aa - ro);
    vec3 ttop = ird*(bb - ro);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 tx = max(tmin.xx, tmin.yz);
    vec2 ty = min(tmax.xx, tmax.yz);
    vec2 tt;
    tt.x = max(tx.x, tx.y);
    tt.y = min(ty.x, ty.y);
    return max(tt, 0.);
}

vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float h = b*b + r*r - dot(ro, ro);
    vec2 tt = h > 0. ? -b - sqrt(h)*vec2(1,-1) : vec2(0);
    return max(tt, 0.);
}

void intersectSplats(
	sampler2D positionAttr, float splatSize, usampler2D indexAttr, uint offset, uint count,
	vec3 rayOrigin, vec3 rayDirection,
	inout BVHIntersectResult res
) {
  res.numLookupsSplats += count;

  for (uint id = 0u; id < count; id++) {
		
    uvec3 indices = uTexelFetch1D( indexAttr, id + offset ).xyz;
		vec3 pos = texelFetch1D( positionAttr, indices.x ).xyz;
    vec2 tt = raySphere(rayOrigin - pos, rayDirection, splatSize);
    float dist = (tt.x + tt.y)/2.; // one sample in the middle of the gaussian splat
    bool inside = tt.x == 0.;

    // raycasting can be restarted from the last sample point, which is inside
    // a splat, so skip points that are inside; this must be fixed, though
		if (!inside && tt.x < tt.y && dist < res.dist[MAX_SPLATS_PER_RAY-1]) {
      
      int j = -1; // insert t into the sorted list

      for (int k = MAX_SPLATS_PER_RAY-1; k >= 0 && dist < res.dist[k]; k--) {
        j = k;

        if (k + 1 < MAX_SPLATS_PER_RAY && res.dist[k] < INFINITY) {
          res.dist[k + 1] = res.dist[k];
          res.splatId[k + 1] = res.splatId[k];
        }
      }

      if (j >= 0) {
        res.dist[j] = dist;
        res.splatId[j] = indices.x;
        res.count = max(res.count, j+1);
      }
    }
	}
}

vec2 rayBVHBox( vec3 rayOrigin, vec3 rayDirection, float splatSize, sampler2D bvhBounds, uint nodeId ) {
	uint cni2 = nodeId * 2u;
	vec3 boundsMin = texelFetch1D( bvhBounds, cni2 + 0u ).xyz - splatSize;
	vec3 boundsMax = texelFetch1D( bvhBounds, cni2 + 1u ).xyz + splatSize;
  return rayBox( rayOrigin, rayDirection, boundsMin, boundsMax );
}

// use a macro to hide the fact that we need to expand the struct into separate fields
#define\
	bvhIntersectSplats(\
		bvh,\
		rayOrigin, rayDirection, splatSize, res\
	)\
	_bvhIntersectSplats(\
		bvh.position, bvh.index, bvh.bvhBounds, bvh.bvhContents,\
		rayOrigin, rayDirection, splatSize, res\
	)

bool _bvhIntersectSplats(
	// bvh info
	sampler2D bvh_position, usampler2D bvh_index, sampler2D bvh_bvhBounds, usampler2D bvh_bvhContents,

	// ray
	vec3 rayOrigin, vec3 rayDirection,
  float splatSize,
	inout BVHIntersectResult res
) {

	int ptr = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

  for (int i = 0; i < MAX_SPLATS_PER_RAY; i++)
	  res.dist[i] = INFINITY;
  res.count = 0;

	while ( ptr >= 0 && ptr < BVH_STACK_DEPTH ) {

		uint nodeId = stack[ ptr-- ];
    res.numLookupsBVH++;
    vec2 tt = rayBVHBox( rayOrigin, rayDirection, splatSize, bvh_bvhBounds, nodeId );
		if (!(tt.x < tt.y) || tt.x > res.dist[MAX_SPLATS_PER_RAY-1])
			continue;

    res.numLookupsBVH++;
		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, nodeId ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;

			intersectSplats(
				bvh_position, splatSize, bvh_index, offset, count,
				rayOrigin, rayDirection, res);

		} else {

			uint leftIndex = nodeId + 1u;
			uint splitAxis = boundsInfo.x & 0x0000ffffu;
			uint rightIndex = boundsInfo.y;

			bool leftToRight = rayDirection[ splitAxis ] >= 0.0;
			uint c1 = leftToRight ? leftIndex : rightIndex;
			uint c2 = leftToRight ? rightIndex : leftIndex;

			stack[ ++ptr ] = c2; // traverse later
			stack[ ++ptr ] = c1; // traverse first
		}
	}

  return res.count > 0;
}
`;
