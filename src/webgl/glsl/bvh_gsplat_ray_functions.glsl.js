export const bvh_gsplat_ray_functions = /* glsl */`

#ifndef MAX_SPLATS_PER_RAY
#define MAX_SPLATS_PER_RAY 1
#endif

#ifndef MAX_SAMPLES_PER_SPLAT
#define MAX_SAMPLES_PER_SPLAT 1
#endif

float[MAX_SPLATS_PER_RAY] gSplatDists;
uint[MAX_SPLATS_PER_RAY] gSplatIds;

struct BVHIntersectResult {
  int numSplats; // 0..MAX_SPLATS_PER_RAY
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
    return tt;
}

vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float h = b*b + r*r - dot(ro, ro);
    return h > 0. ? -b - sqrt(h)*vec2(1,-1) : vec2(0);
}

void intersectSplats(
	sampler2D positionAttr, sampler2D splatSizes, usampler2D indexAttr, uint offset, uint count,
	vec3 rayOrigin, vec3 rayDirection,
	inout BVHIntersectResult res
) {
  res.numLookupsSplats += count;

  for (uint id = 0u; id < count; id++) {
		
    uint splatId = uTexelFetch1D( indexAttr, id + offset ).x;
		vec3 pos = texelFetch1D( positionAttr, splatId ).xyz;
    float radius = texelFetch1D( splatSizes, splatId ).x;
    vec2 tt = raySphere(rayOrigin - pos, rayDirection, radius);
    
    if (tt.x < tt.y) {
      for (int s = 0; s < MAX_SAMPLES_PER_SPLAT; s++) {
        float dist = mix(tt.x, tt.y, (float(s) + 0.5)/float(MAX_SAMPLES_PER_SPLAT));

        if (dist > 0. && dist < gSplatDists[MAX_SPLATS_PER_RAY-1]) {
          // insert the new sample point into the sorted list
          res.numSplats = min(res.numSplats + 1, MAX_SPLATS_PER_RAY);

          for (int k = res.numSplats - 1; k >= 0 && dist < gSplatDists[k]; k--) {
            if (k + 1 < MAX_SPLATS_PER_RAY) {
              gSplatDists[k + 1] = gSplatDists[k];
              gSplatIds[k + 1] = gSplatIds[k];
            }

            gSplatDists[k] = dist;
            gSplatIds[k] = splatId;
          }
        }
      }
    }
	}
}

vec2 rayBVH( vec3 rayOrigin, vec3 rayDirection, sampler2D bvhBounds, uint nodeId ) {
	uint cni2 = nodeId * 2u;
	vec3 boundsMin = texelFetch1D( bvhBounds, cni2 + 0u ).xyz;
	vec3 boundsMax = texelFetch1D( bvhBounds, cni2 + 1u ).xyz;
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
  sampler2D splatSizes,
	inout BVHIntersectResult res
) {

	int ptr = 0;
	uint stack[ BVH_STACK_DEPTH ];
	stack[ 0 ] = 0u;

  for (int i = 0; i < MAX_SPLATS_PER_RAY; i++)
	  gSplatDists[i] = INFINITY;
  res.numSplats = 0;

	while ( ptr >= 0 && ptr < BVH_STACK_DEPTH ) {

		uint nodeId = stack[ ptr-- ];
    res.numLookupsBVH++;
    vec2 tt = rayBVH( rayOrigin, rayDirection, bvh_bvhBounds, nodeId );
		if (tt.x >= tt.y || tt.x >= gSplatDists[MAX_SPLATS_PER_RAY-1])
			continue;

    res.numLookupsBVH++;
		uvec2 boundsInfo = uTexelFetch1D( bvh_bvhContents, nodeId ).xy;
		bool isLeaf = bool( boundsInfo.x & 0xffff0000u );

		if ( isLeaf ) {

			uint count = boundsInfo.x & 0x0000ffffu;
			uint offset = boundsInfo.y;

			intersectSplats(
				bvh_position, splatSizes, bvh_index, offset, count,
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

  return res.numSplats > 0;
}
`;
