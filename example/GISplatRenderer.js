import * as THREE from 'three';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

import {
  MeshBVH,
  MeshBVHHelper,
  BVHShaderGLSL,
  MeshBVHUniformStruct,
  FloatVertexAttributeTexture
} from 'three-mesh-bvh';

const SHADOW_MAP_SIZE = 2048;

let bvh, bvhGeometry, bvhHelper;
let nextSplatPass, raymarchingPass, canvasDrawPass;
let splatColorsRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
let shadowMapRT = new THREE.WebGLArrayRenderTarget(1, 1, 1, { type: THREE.UnsignedShortType, format: THREE.RedFormat });

export class GSplatsDataUniformStruct {
  constructor(params) {
    this.splatsCount = bvhGeometry.attributes.position.count; // (xyz, radius) x N
    this.maxStdDev = 2 ** params.maxStdDev;
    this.splatOpacity = 2 ** params.splatOpacity;
    this.ambientLight = 2 ** params.ambientLight;
    this.fogDensity = params.fogDensity > -3 ? 10 ** params.fogDensity : 0;
    this.shadowMap = shadowMapRT.texture;
    this.splatColors = splatColorsRT.texture;
  }
}

export class DoubleBufferRenderTarget {
  rtA = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });
  rtB = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });

  swap() {
    [this.rtA, this.rtB] = [this.rtB, this.rtA];
  }

  setSize(w, h) {
    this.rtA.setSize(w, h);
    this.rtB.setSize(w, h);
  }
}

THREE.ShaderChunk['yuv_rgb'] = /* glsl */`
  const mat3 YUV_RGB = transpose(mat3(1,1,1,  0,-0.34,1.77, 1.4,-0.72,0));
  const mat3 RGB_YUV = inverse(YUV_RGB); // yuv = rgb * RGB_YUV
`;

THREE.ShaderChunk['gsplats_data'] = /* glsl */`
  #define USE_GAMMA 1 // blend RGB^2, then output sqrt(RGB)

  struct GSplatsData {
    int splatsCount;
    float maxStdDev;
    float splatOpacity;
    float ambientLight;
    float fogDensity;
    
    sampler2D splatColors;
    sampler2DArray shadowMap;
  };
`;

THREE.ShaderChunk['unpack_4x16'] = /* glsl */`
  #define pack2(xy) uintBitsToFloat(packHalf2x16(vec2(xy)))
  #define unpack2(f32) unpackHalf2x16(floatBitsToUint(float(f32)))
  // float 0..1 <-> uint 0..65535
  #define PACK_2x16(xy)   uintBitsToFloat(packUnorm2x16(vec2(xy)))
  #define PACK_4x16(v)    vec2(PACK_2x16(v.xy), PACK_2x16(v.zw))
  #define UNPACK_2x16(x)  unpackUnorm2x16(floatBitsToUint(x))
  #define UNPACK_4x16(v)  vec4(UNPACK_2x16(v.x), UNPACK_2x16(v.y))
`;

THREE.ShaderChunk['pack_pixel_data'] = /* glsl */`
  struct PixelData { 
    vec4 color; // .w < 1.0 - the accumulated density
    float zDepth; 
    int cost; 
  };

  vec4 packPixelData(PixelData pd) {
    vec3 yuv = pd.color.rgb * RGB_YUV;
    vec4 pixel;
    pixel.x = yuv.x; // Y' of Y'UV
    pixel.y = pack2(yuv.yz); // UV of Y'UV, as 2 x float16
    pixel.z = pd.zDepth;
    pixel.w = PACK_2x16(vec2(pd.color.w, float(pd.cost) / float(0xFFFF)));
    return pixel;
  }

  PixelData unpackPixelData(vec4 pixel) {
    vec3 yuv;
    yuv.x = pixel.x;
    yuv.yz = unpack2(pixel.y);
    vec2 wc = UNPACK_2x16(pixel.w);
    PixelData pd;
    pd.color = vec4(yuv * YUV_RGB, wc.x);
    pd.zDepth = pixel.z;
    pd.cost = int(wc.y * float(0xFFFF));
    return pd;
  }
`;

THREE.ShaderChunk['gaussian_utils'] = /* glsl */`
  const float SQRT_PI = sqrt(radians(180.));
  const float SQRT_2 = sqrt(2.0);

  float gaussian3d(vec3 r) {
    return exp(-dot(r, r));
  }
  
  // integrate( exp(-x*x))*2/sqrt(PI), 0..x ) = -1..1
  // https://en.wikipedia.org/wiki/Error_function
  float erf(float x) {
    if (abs(x) > 3.5)
        return sign(x);

    return sign(x)*sqrt(1. - exp2(-SQRT_PI*x*x));
  }

  // 2/sqrt(PI) * integrate( exp(-|pos + dir*t|^2), t=0..len )
  // https://en.wikipedia.org/wiki/Gaussian_integral
  float erf3d(vec3 pos, vec3 dir, float len) {
    if (len < 0.01)
      return len*gaussian3d(pos + dir*len*0.5);

    float b = dot(pos, dir);          // -INF..INF
    float h = dot(pos, pos) - b*b;    // 0..INF
    float s = erf(b + len) - erf(b);  // 0..2
    return exp(-h)*s*SQRT_PI*0.5;     // 0..sqrt(PI)
  }

  // integrate( exp(-(pos + dir*t).y), t=0..len )
  // https://iquilezles.org/articles/fog
  float expfog_3d(vec3 pos, vec3 dir, vec3 up, float len) {
    float d = dot(dir, up);
    float p = dot(pos, up);

    if (abs(len * d) < 0.01)
      return exp(-p) * len;

    return exp(-p) * (1.0 - exp(-len * d)) / d;
  }
`;

THREE.ShaderChunk['ray_utils'] = /* glsl */`
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
`;

// Uses BVH to compute aggregate density and shadow at the current spot.
THREE.ShaderChunk['bvh_shadows_raycasting'] = /* glsl */`
  mat2x3 bvhBounds = mat2x3(0); // min..max or AABB

  // inputs
  vec3 bvhRayDir;
  vec3 gPos = vec3(0);
  vec3 gRayDir = vec3(0);
  vec3 gSunPos = vec3(0);
  vec3 gSunDir = vec3(0);
  float gSunDist = 0.;
  vec4 gFogSplat = vec4(0);
  vec4 gFogColor = vec4(0);
  int gMinShadowMapLayer = 0;
  
  // outputs
  float gSumShadow = 0.;
  vec4 gSumColor = vec4(0);
  int gShadowMapLayers = 0;
  int gTexLookups = 0;

  void initFogSplat(mat4 modelWorldMatrix) {
    gFogSplat = vec4(0, -3, 0, gsd.maxStdDev);
    gFogColor = vec4(1, 1, 1, gsd.fogDensity/gsd.splatOpacity);
    // (aa, bb) is in sun coords, so vec3(0,0,1) points to the sun
    gFogSplat.xyz = (vec4(gFogSplat.xyz, 1) * inverse(modelWorldMatrix)).xyz;
  }

  void bvhInitBounds() {
    vec3 aa = texelFetch1D( bvh.bvhBounds, 0u ).xyz;
    vec3 bb = texelFetch1D( bvh.bvhBounds, 1u ).xyz;
    bvhBounds = mat2x3(aa, bb);
    gTexLookups++;
  }

  vec4 integrateSplat(vec4 splat, vec4 color, vec3 pos, vec3 dir, float len) {
    if (splat.w < 1e-6)
      return vec4(0);
    
    color.w *= gsd.splatOpacity;
    
    #if NEED_COLOR
      #if USE_GAMMA
        color.rgb *= color.rgb;
      #endif
      color.rgb *= color.w;
    #endif

    float scale = SQRT_2 / gsd.maxStdDev * splat.w;
    float weight = scale * erf3d((pos - splat.xyz)/scale, dir, len/scale);

    // The proper integral would be:
    //
    //    erf3d((pos - splat.xyz)/splat.w/scale, dir)*splat.w*scale
    //
    // However rasterizers implicitly multiply opacity of splats
    // by their size, so the splat.w multiplier is omitted here.    
    return weight/splat.w * color;
  }

  vec4 getShadowMapUV() {
    vec3 aa = bvhBounds[0], bb = bvhBounds[1];
    vec3 p = (gPos - aa) / (bb - aa); // 0..1 x 0..1 x -INF..1
    float numLayers = float(gShadowMapLayers);
    float layer = max(ceil(numLayers * p.z), float(gMinShadowMapLayer));
    float delta = layer/numLayers - p.z; // 0..1
    vec3 sun = gSunDir / (bb - aa); // gSunDir.z = 1.0 + eps
    vec2 uv = p.xy + sun.xy/sun.z * delta; // uv.z = layer/numLayers
    float dist = delta * (bb.z - aa.z)/gSunDir.z;
    return vec4(uv, layer, dist);
  }

  void bvhInitSearch() {
    gSumShadow = 0.;
    gSumColor = vec4(0);
    gTexLookups = 0;
    gShadowMapLayers = textureSize(gsd.shadowMap, 0).z;

    #if NEED_SHADOW
      vec4 uv = getShadowMapUV();

      if (int(uv.z) < gShadowMapLayers) {
        gSumShadow += 8.0*texture(gsd.shadowMap, uv.xyz).x;
        gTexLookups++;
        gSunDist = uv.w;
      }
    #endif

    #if NEED_COLOR
      gSumColor = integrateSplat(gFogSplat, gFogColor, gPos - gRayDir*RAY_STEP*0.5, gRayDir, RAY_STEP);
    #endif
  }

  bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
    #if NEED_SHADOW
    
      vec2 tt = rayBox( gPos, gSunDir, boundsMin, boundsMax );
      tt.x = max(tt.x, 0.);
      tt.y = min(tt.y, gSunDist);
      return tt.x < tt.y;

    #else 

      return gPos == clamp( gPos, boundsMin, boundsMax );

    #endif
  }

  bool bvhVisitSplat(uint splatId) {
    vec4 splat = texelFetch1D( bvh.position, splatId );
    
    if (splat.w < 1e-6)
      return false;

    vec3 r = (gPos - splat.xyz) / splat.w;
    float rd = clamp(dot(r, -gSunDir), 0., gSunDist/splat.w);
    float r2 = dot(r, r);
    float h2 = r2 - rd*rd; // h2 < r2
    vec4 color = vec4(0);

    if (bool(NEED_SHADOW) && h2 < 1. || bool(NEED_COLOR) && r2 < 1.)
      color = texelFetch1D(gsd.splatColors, splatId);

    // see if the sunray intersects the splat
    #if NEED_SHADOW
      if (h2 < 1.) {
        gSumShadow += integrateSplat(splat, color, gPos, gSunDir, gSunDist).w;
      }
    #endif

    // see if the current pos is inside the splat
    #if NEED_COLOR
      if (r2 < 1.) {
        // Rasterizers render small splats with the same opacity as large splats,
        // but if the splats were to be integrated properly, the opacity would have
        // to be scaled by the splat size: integrate(exp(-1/2 * |r/s|^2)) = sqrt(2*PI)*s
        // This means that rasterizers implicitly scale the density of splats and this
        // must to be accounted for here.
        gSumColor += integrateSplat(splat, color, gPos - gRayDir*RAY_STEP*0.5, gRayDir, RAY_STEP);
      }
    #endif

    return color.w > 0.;
  }
`;

// Find the next splat starting from rayOrigin + rayDir*abs(pixelData.z).
// It's a no-op if pixelData.z > 0. because there is a splat at that point. 
export class NextSplatMaterial extends THREE.ShaderMaterial {

  updateDefines(params) {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.RAY_STEP = 10 ** params.rayStep;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      defines: {

        RAY_STEP: 0.001,
        BVH_STACK_DEPTH: 64,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelWorldMatrix: { value: new THREE.Matrix4() },

        frameId: { value: 0 },

      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;
        uniform sampler2D pixelData;
        uniform int frameId;
        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelWorldMatrix;

        #include <ray_utils>
        #include <gaussian_utils>

        vec3 bvhRayDir;
        struct BVHRay { vec3 origin; } bvhRay;
        struct BVHNearest { float dist; uint splatId; } bvhNearest;

        void bvhInitSearch() {
          bvhNearest = BVHNearest(INFINITY, 0u);
        }

        bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
          vec2 tt = rayBox( bvhRay.origin, bvhRayDir, boundsMin, boundsMax );
          return tt.x < tt.y && tt.x < bvhNearest.dist && tt.y > 0.;
        }

        // Finds the nearest splat along the ray.
        bool bvhVisitSplat(uint splatId) {
          vec4 splat = texelFetch1D( bvh.position, splatId );
          vec2 tt = raySphere(bvhRay.origin - splat.xyz, bvhRayDir, splat.w);
          tt = max(tt, vec2(0));

          if (tt.x < tt.y && tt.x < bvhNearest.dist && tt.y > 0.) {
            bvhNearest.dist = tt.x;
            bvhNearest.splatId = splatId;
            return true;
          }
          
          return false;
        }

        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        #include <common>
        #include <unpack_4x16>

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDir;
          ndcToCameraRay(
            ndc, inverse(modelWorldMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDir);
          rayDir = normalize(rayDir);

          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          if (frameId == 0) {
            rayData.z = -1e-6;
            rayData.w = PACK_2x16(0);
          }

          if (rayData.z < 0. && abs(rayData.z) < INFINITY) {
            bvhRayDir = rayDir;
            bvhRay.origin = rayOrigin + rayDir * abs(rayData.z);
            
            bvhSearchSplats( bvh );
            
            rayData.z = abs(rayData.z) + bvhNearest.dist + RAY_STEP*0.5;
            
            vec2 wc = UNPACK_2x16(rayData.w);
            wc.y += float(bvhTexLookups)/float(0xFFFF);
            rayData.w = PACK_2x16(wc);
          }

          gl_FragColor = rayData;
        }`
    });
  }
}

// Computes density and shadows at the current point and makes a RAY_STEP forward.
// It's a no-op if pixelData.z < 0. because there are no splats at the current point.
export class RaymarchingMaterial extends THREE.ShaderMaterial {

  updateDefines(params) {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.RAY_STEP = 10 ** params.rayStep;
    this.defines.NEED_SHADOW = +params.shadows;
    this.needsUpdate = true;
  }

  constructor(params) {

    super({

      defines: {

        RAY_STEP: 0.001,
        BVH_STACK_DEPTH: 64,
        INTEGRATE_FOG: 256,
        NEED_SHADOW: 1,
        NEED_COLOR: 1,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelWorldMatrix: { value: new THREE.Matrix4() },

        lightPos: { value: params.lightPos },
        frameId: { value: 0 },

      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;
        uniform sampler2D pixelData;
        uniform int frameId;
        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelWorldMatrix;
        uniform vec3 lightPos;

        #include <yuv_rgb>
        #include <ray_utils>
        #include <gaussian_utils>
        #include <bvh_shadows_raycasting>
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        #include <common>
        #include <unpack_4x16>
        #include <pack_pixel_data>

        // Fast fog integration outside the AABB bounding box.
        vec4 integrateFog(vec3 pos, vec3 dir, float len) {
          vec4 sum = vec4(0);

          for (int i = 0; i < INTEGRATE_FOG; i++) {
            float dt = len / float(INTEGRATE_FOG);
            float t = float(i)*dt;
            vec3 p = pos + dir*t;
            vec3 sunDir = gSunPos - p;
            float sunDist = length(sunDir);
            sunDir /= sunDist;

            float lum = 1.0;

            #if NEED_SHADOW
              float weight = integrateSplat(gFogSplat, gFogColor, p, sunDir, sunDist).w;
              lum *= exp(-weight);
              lum += gsd.ambientLight; // ambient occlusion (AO) or global illumination (GI)
            #endif

            vec4 vol = integrateSplat(gFogSplat, gFogColor, p, dir, dt);
            vol.rgb *= lum;
            
            sum.rgb += exp(-sum.w) * vol.rgb;
            sum.w += vol.w;
          }

          return sum;
        }

        // Fog outside the AABB can be integrated quickly since no BVH lookups are necessary.
        bool addExteriorFog(vec3 rayOrigin, vec3 rayDir, inout float zDepth, inout vec4 color) {
          vec3 aa = bvhBounds[0];
          vec3 bb = bvhBounds[1];

          aa.z -= (bb.z - aa.z)*15.; // the shadow that the AABB box casts

          // rayOrigin doesn't change, but zDepth does
          vec2 tt = rayBox(rayOrigin, rayDir, aa, bb);
          tt.x = max(tt.x, 0.);
          
          if (tt.x >= tt.y)
            tt = vec2(INFINITY);

          float maxFogDist = length(aa - bb);

          if (frameId == 0) {
            zDepth = tt.x + RAY_STEP*0.5;
            
            if (zDepth > RAY_STEP) {
              #if INTEGRATE_FOG
                // add fog that's in front of the AABB
                float len = min(zDepth, maxFogDist);
                vec3 pos = rayOrigin;
                if (zDepth < INFINITY)
                  pos += rayDir*(zDepth - len);
                vec4 fog = integrateFog(pos, rayDir, len);
                color.rgb += exp(-color.w) * fog.rgb;
                color.w += fog.w;
              #endif

              return true;
            }
          } else if (zDepth > tt.y) {
            #if INTEGRATE_FOG
              // add fog that's behind the AABB
              vec4 fog = integrateFog(rayOrigin + zDepth*rayDir, rayDir, maxFogDist);
              color.rgb += exp(-color.w) * fog.rgb;
              color.w += fog.w;
            #endif

            zDepth = INFINITY;
            return true;
          }

          return false;
        }

        // The shadow map can be used to skip empty interior areas.
        bool isAreaEmpty() {
          #if !NEED_SHADOW
            return false; // no shadow map
          #endif

          vec4 uv = getShadowMapUV();
          float above = 0.0, below = 1.0;

          if (int(uv.z) == clamp(int(uv.z), 1, gShadowMapLayers - 1)) {
            above = 8.0*texture(gsd.shadowMap, uv.xyz).x;
            below = 8.0*texture(gsd.shadowMap, uv.xyz - vec3(0,0,1)).x;
            gTexLookups += 2;
          }
          
          return abs(above - below) < 0.01;
        }

        // This applies to points inside the AABB where BVH lookups are necessary.
        bool blendSplats(inout PixelData pd) {
          //if (isAreaEmpty())
          //  return true;

          bvhSearchSplats( bvh );
          pd.cost += bvhTexLookups + gTexLookups;
          if (gSumColor.w <= 0. && gFogSplat.w == 0.)
            return false;

          float luminance = 1.0;

          #if NEED_SHADOW
            float dist = length(gSunPos - gPos); // shadowMap doesn't include fog
            gSumShadow += integrateSplat(gFogSplat, gFogColor, gPos, gSunDir, dist).w;
            luminance *= exp(-gSumShadow);
            luminance += gsd.ambientLight; // ambient occlusion (AO) or global illumination (GI)
          #endif

          vec4 vol = gSumColor;
          vol.rgb *= luminance;
          
          pd.color.rgb += exp(-pd.color.w) * vol.rgb;
          pd.color.w += vol.w;
          return true;
        }

        vec4 compress(PixelData pd) {
          pd.color.w /= 8.;
          if (pd.color.w >= 1.0)
              pd.zDepth = INFINITY;
          return packPixelData(pd);
        }

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDir;
          ndcToCameraRay(
            ndc, inverse(modelWorldMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDir);
          rayDir = normalize(rayDir);

          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          if (frameId == 0) {
            PixelData pd = unpackPixelData(rayData);
            pd.color = vec4(0);
            // NextSplatMaterial runs first when fog=0
            if (gsd.fogDensity > 0.)
              pd.zDepth = 0., pd.cost = 0;
            rayData = packPixelData(pd);
          }

          if (rayData.z < 0.)
            discard; // it's NextSplatMaterial's turn

          // skip already rendered pixels
          if (rayData.z >= INFINITY) {
            gl_FragColor = rayData;
            return;
          }

          if (gsd.fogDensity > 0.)
            initFogSplat(modelWorldMatrix);

          gRayDir = rayDir;
          gPos = rayOrigin + rayDir * rayData.z;
          gSunPos = (vec4(lightPos, 1) * inverse(modelWorldMatrix)).xyz;
          gSunDir = normalize(gSunPos - gPos);
          gSunDist = length(gSunPos - gPos);
          bvhRayDir = gSunDir;

          bvhInitBounds();

          PixelData pd = unpackPixelData(rayData);
          pd.color.w *= 8.; // density range: exp(0)..exp(-8) = 1..0.0003

          if (gsd.fogDensity > 0. && addExteriorFog(rayOrigin, rayDir, pd.zDepth, pd.color)) {
            // nothing to do
          } else if (blendSplats(pd)) {
            pd.zDepth += RAY_STEP;
          } else {
            pd.zDepth *= -1.; // tell NextSplatMaterial to find the next splat
          }
          
          gl_FragColor = compress(pd);
        }`
    });
  }
}

export class ShadowMapMaterial extends THREE.ShaderMaterial {

  updateDefines(params) {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.needsUpdate = true;
  }

  constructor(params) {

    super({

      defines: {

        BVH_STACK_DEPTH: 64,
        NEED_SHADOW: 1,
        NEED_COLOR: 0,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        modelWorldMatrix: { value: new THREE.Matrix4() },
        lightPos: { value: params.lightPos },
        uLayer: { value: 0 },

      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;
        uniform mat4 modelWorldMatrix;
        uniform vec3 lightPos;
        uniform int uLayer;

        #include <yuv_rgb>
        #include <ray_utils>
        #include <gaussian_utils>
        #include <bvh_shadows_raycasting>
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        #include <common>
        #include <unpack_4x16>

        void main() {
          int numLayers = textureSize(gsd.shadowMap, 0).z;

          if (gsd.fogDensity > 0.)
            initFogSplat(modelWorldMatrix);

          bvhInitBounds();

          vec3 aa = bvhBounds[0];
          vec3 bb = bvhBounds[1];

          gMinShadowMapLayer = uLayer + 1;
          gPos = mix(aa, bb, vec3(vUv, float(uLayer)/float(numLayers)));
          gSunPos = (vec4(lightPos, 1) * inverse(modelWorldMatrix)).xyz;
          gSunDir = normalize(gSunPos - gPos);
          gSunDist = length(gSunPos - gPos);
          gRayDir = gSunDir;
          bvhRayDir = gSunDir;

          bvhSearchSplats( bvh );
          
          gl_FragColor.x = gSumShadow/8.0; // gFogSplat not included
        }`
    });
  }
}

// (f_dc, rgb, opacity) -> rgba
export class SplatColorsMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        ply: { value: null },
      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        uniform struct { sampler2D f_dc, rgb, opacity; } ply;

        #include <common>

        void main() {
          vec2 size = vec2(textureSize(ply.f_dc, 0));

          vec3 rgb = texelFetch(ply.rgb, ivec2(vUv*size), 0).rgb;
          vec3 f_dc = texelFetch(ply.f_dc, ivec2(vUv*size), 0).rgb;
          float opacity = texelFetch(ply.opacity, ivec2(vUv*size), 0).x;

          gl_FragColor = vec4(1);

          if (!isnan(rgb.x)) gl_FragColor.rgb = rgb/255.;
          if (!isnan(f_dc.x)) gl_FragColor.rgb = f_dc/sqrt(PI)*0.5 + 0.5;
          if (!isnan(opacity)) gl_FragColor.a = 1./(1. + exp(-opacity));
        }`
    });
  }
}

export class CanvasDrawMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        brightness: { value: 1 },
        monochrome: { value: false },
        pixelData: { value: null },
        frameId: { value: 0 },
        gsd: { value: null },
      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        uniform sampler2D pixelData;
        uniform int frameId;
        uniform bool monochrome;
        uniform float brightness;

        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <gsplats_data>
        #include <pack_pixel_data>

        ${BVHShaderGLSL.common_functions}

        uniform GSplatsData gsd;
        vec2 size;
        const int M = 10;

        float vmax3(vec3 v) { return max(max(v.x, v.y), v.z); }
        float vmin3(vec3 v) { return -vmax3(-v); }

        void drawShadowMap(inout vec4 o) {
          vec2 uv = vUv*float(M) - vec2(0,M-1);
          if (uv != clamp(uv, 0., 1.))
            return;

          int numLayers = textureSize(gsd.shadowMap, 0).z;
          int layer = frameId/60 % numLayers;
          float shadow = 8.0*texture(gsd.shadowMap, vec3(uv, layer)).x;
          o.rgb = vec3(1,3,9) * (1. - exp(-shadow));
        }

        void drawProgress(inout vec4 o) {
          vec2 uv = vUv*float(M);
          if (uv != clamp(uv, 0., 1.))
            return;

          vec4 pixel = texelFetch(pixelData, ivec2(uv*size), 0);
          PixelData pd = unpackPixelData(pixel);

          if (pd.zDepth < INFINITY) {
            float weight = 8.0*pd.color.w;
            o.rgb = vec3(3,1,9) * exp(-weight);
          }
        }

        void drawCost(inout vec4 o) {
          vec2 uv = vUv*float(M) - vec2(M-1,0);
          if (uv != clamp(uv, 0., 1.))
            return;

          vec4 pixel = texture(pixelData, uv);
          PixelData pd = unpackPixelData(pixel);
          o.rgb = vec3(9,3,1) * float(pd.cost)/float(0x10000);
        }

        void main() {
          size = vec2(textureSize(pixelData, 0));
          vec4 pixel = texelFetch(pixelData, ivec2(vUv*size), 0);
          PixelData pd = unpackPixelData(pixel);
          vec4 o = pd.color;

          #if USE_GAMMA
            // RGB <-> YUV may create negative RGB values
            o.rgb = sqrt(max(o.rgb, vec3(0)));
          #endif

          o.rgb *= brightness;

          if (monochrome)
            o.rgb = (vmin3(o.rgb) + vmax3(o.rgb)) * vec3(0.5);

          //o.rgb += exp(-o.w) * backgroundRGB;

          //drawShadowMap(o);
          drawProgress(o);
          drawCost(o);

          if (isnan(dot(o, vec4(1))))
            o = vec4(0,1,0,1);

          gl_FragColor = vec4(o.rgb, 1);
        }`
    });
  }
}

export function updateShadowMapGI(renderer, params, bvh, gsd, pointCloud) {
  console.time('Update shadowMap');
  let size = SHADOW_MAP_SIZE, layers = params.shadowMapLayers;
  shadowMapRT.setSize(size, size, layers);

  let layerRT = new THREE.WebGLRenderTarget(size, size, { type: THREE.FloatType, format: THREE.RedFormat });
  let copy = new FullScreenQuad(new THREE.ShaderMaterial(CopyShader));
  copy.material.uniforms.tDiffuse.value = layerRT.texture;

  let shadowMapPass = new FullScreenQuad(new ShadowMapMaterial(params));
  shadowMapPass.material.updateDefines(params);

  let u = shadowMapPass.material.uniforms;
  u.bvh.value.updateFrom(bvh);
  u.gsd.value = gsd;
  pointCloud.updateMatrixWorld();
  u.modelWorldMatrix.value.copy(pointCloud.matrixWorld);

  for (let i = layers - 1; i >= 0; i--) {
    u.uLayer.value = i;
    renderer.setRenderTarget(layerRT);
    shadowMapPass.render(renderer);

    renderer.setRenderTarget(shadowMapRT, i);
    copy.render(renderer);

    // No idea why this isn't working.
    //renderer.copyTextureToTexture(
    //  layerRT.texture, shadowMapRT.texture,
    //  null, new THREE.Vector3(0, 0, i));
  }

  shadowMapPass.dispose();
  layerRT.dispose();
  copy.dispose();

  // Read data to CPU to measure time correctly.
  // This wouldn't work with a 3D texture.
  renderer.readRenderTargetPixels(shadowMapRT,
    0, 0, 1, 1, new Uint8Array(4), 0);
  renderer.setRenderTarget(null);
  console.timeEnd('Update shadowMap');
}

export async function loadPLY(url) {
  let ply = new PLYLoader();

  // these will go to geometry.attributes
  ply.setCustomPropertyNameMapping({
    scale: ['scale_0', 'scale_1', 'scale_2'], // scale = log(S)
    f_dc: ['f_dc_0', 'f_dc_1', 'f_dc_2'], // f_dc = (RGB - 0.5)*sqrt(PI)*2.0
    rgb: ['red', 'green', 'blue'], // 0..255
    opacity: ['opacity'], // opacity = -log(1.0/A - 1.0), A=0..1
    // rot: ['rot_0', 'rot_1', 'rot_2', 'rot_3'], // quaternion rotation
  });

  return await ply.loadAsync(url);
}

export function updateSplatColors(renderer, pointCloud) {
  // this is computed once when the splats file is loaded
  let attributes = pointCloud.geometry.attributes;
  let numSplats = attributes.position.count;

  let rgb = new FloatVertexAttributeTexture();
  let f_dc = new FloatVertexAttributeTexture();
  let opacity = new FloatVertexAttributeTexture();

  let attrNAN = new THREE.BufferAttribute(new Float32Array(numSplats), 1);
  attrNAN.array.fill(Number.NaN);

  rgb.updateFrom(attributes.rgb || attrNAN); // 0..255, uint8
  f_dc.updateFrom(attributes.f_dc || attrNAN);
  opacity.updateFrom(attributes.opacity || attrNAN);

  let { width, height } = opacity.image;
  splatColorsRT.setSize(width, height);

  let shader = new FullScreenQuad(new SplatColorsMaterial());
  shader.material.uniforms.ply.value = { f_dc, rgb, opacity };
  renderer.setRenderTarget(splatColorsRT);
  shader.render(renderer);

  f_dc.dispose();
  opacity.dispose();
  shader.dispose();
}

export function getSunMatrix4(zAxis) {
  let c = zAxis.clone().normalize();
  let b = Math.abs(c.x) > Math.abs(c.z) ?
    new THREE.Vector3(-c.y, c.x, 0).normalize() :
    new THREE.Vector3(0, -c.z, c.y).normalize();
  let a = c.clone().cross(b);

  return new THREE.Matrix4(
    a.x, a.y, a.z, 0,
    b.x, b.y, b.z, 0,
    c.x, c.y, c.z, 0,
    0, 0, 0, 1);
}

export function disposeBVH() {
  bvh = null;
  bvhGeometry = null;
}

export function updateBVH(params, pointCloud, scene) {
  console.time('updateBVH');

  let attributes = pointCloud.geometry.attributes;

  let m = 1 << params.sparsity;
  let position = attributes.position;
  let numSplats = position.count;
  let numSplatsM = numSplats / m | 0;

  let position3 = new THREE.BufferAttribute(new Float32Array(numSplats * 9), 3); // [xyz, xyz - r, xyz + r]
  let position4 = new THREE.BufferAttribute(new Float32Array(numSplats * 4), 4); // (xyz, radius)
  let baseScale = 2 ** (params.maxStdDev + params.splatScale);
  let defaultScale = attributes.scale && Number.isFinite(attributes.scale.array[0]) ? 0 : 0.0025;

  for (let i = 0; i < numSplats; i++) {
    let x = position.array[i * m * 3 + 0];
    let y = position.array[i * m * 3 + 1];
    let z = position.array[i * m * 3 + 2];
    let r = baseScale * (defaultScale || Math.exp(attributes.scale.array[i * m * 3]));

    // this is for GLSL shaders

    position4.array[i * 4 + 0] = x;
    position4.array[i * 4 + 1] = y;
    position4.array[i * 4 + 2] = z;
    position4.array[i * 4 + 3] = r;

    // this is for MeshBVH

    position3.array[i * 9 + 0] = x;
    position3.array[i * 9 + 1] = y;
    position3.array[i * 9 + 2] = z;

    position3.array[i * 9 + 3] = x - r;
    position3.array[i * 9 + 4] = y - r;
    position3.array[i * 9 + 5] = z - r;

    position3.array[i * 9 + 6] = x + r;
    position3.array[i * 9 + 7] = y + r;
    position3.array[i * 9 + 8] = z + r;
  }

  //console.log('max(position3)', position3.array.reduce((s, x) => Math.max(s, Math.abs(x)), 0));
  //console.log('max(position4)', position4.array.reduce((s, x) => Math.max(s, Math.abs(x)), 0));

  let index = [];

  for (let i = 0; i < numSplatsM; i++) {
    let j = i * m * 3;
    index.push(j + 0, j + 1, j + 2);
  }

  if (!bvhGeometry) {
    bvhGeometry = new THREE.BufferGeometry();
    bvhGeometry.setIndex(index);
    bvhGeometry.setAttribute('position', position3);
    bvhGeometry.computeBoundsTree(params.bvhOptions);
  } else {
    bvhGeometry.attributes.position.copy(position3);
    bvhGeometry.boundsTree.refit();
  }

  // BVH must be aligned with sunrays for best performance
  let bvhHelperMesh = new THREE.Mesh(bvhGeometry, new THREE.MeshBasicMaterial());
  bvhHelperMesh.matrix = pointCloud.matrix.clone();
  bvhHelperMesh.matrixAutoUpdate = false;

  scene.remove(bvhHelper);
  bvhHelper = new MeshBVHHelper(bvhHelperMesh, params.depth);
  scene.add(bvhHelper);
  bvhHelper.displayParents = true;
  bvhHelper.opacity = 0.1;
  bvhHelper.update();

  if (!bvh) {
    bvh = new MeshBVH(bvhGeometry, params.bvhOptions);
  } else {
    bvh.refit();
  }

  // GLSL needs 4-element position attr for efficiency, but MeshBVH doesn't support that,
  // so build the BVH first, and then replace the position attr, as MeshBVH no longer needs it.
  bvhGeometry.attributes.position.copy(position4);
  position3 = null; // it's been replaced with position4

  console.timeEnd('updateBVH');

  return bvh;
}

export function runRaymarchingPass(renderer, camera, pointCloud, params, frameId, pixelsRT) {
  let gsd = new GSplatsDataUniformStruct(params);

  if (!gsd.fogDensity) {
    if (!nextSplatPass) {
      nextSplatPass = new FullScreenQuad(new NextSplatMaterial());
      nextSplatPass.material.updateDefines(params);
    }

    let u = nextSplatPass.material.uniforms;
    u.bvh.value.updateFrom(bvh);
    u.gsd.value = gsd;
    u.pixelData.value = pixelsRT.rtA.texture;
    u.frameId.value = frameId;
    u.cameraWorldMatrix.value.copy(camera.matrixWorld);
    u.projectionMatrix.value.copy(camera.projectionMatrix);
    u.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
    renderer.setRenderTarget(pixelsRT.rtB);
    nextSplatPass.render(renderer);
    pixelsRT.swap();
  }

  if (!raymarchingPass) {
    raymarchingPass = new FullScreenQuad(new RaymarchingMaterial(params));
    raymarchingPass.material.updateDefines(params);
  }

  let u = raymarchingPass.material.uniforms;
  u.bvh.value.updateFrom(bvh);
  u.gsd.value = gsd;
  u.pixelData.value = pixelsRT.rtA.texture;
  u.frameId.value = frameId;
  u.cameraWorldMatrix.value.copy(camera.matrixWorld);
  u.projectionMatrix.value.copy(camera.projectionMatrix);
  u.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
  renderer.setRenderTarget(pixelsRT.rtB);
  raymarchingPass.render(renderer);
  pixelsRT.swap();
}

export function runRenderPass(renderer, params, frameId, pixelsRT) {
  if (!canvasDrawPass) {
    canvasDrawPass = new FullScreenQuad(new CanvasDrawMaterial());
  }

  let gsd = new GSplatsDataUniformStruct(params);
  let u = canvasDrawPass.material.uniforms;
  u.brightness.value = 2 ** params.brightness;
  u.monochrome.value = params.monochrome;
  u.frameId.value = frameId;
  u.pixelData.value = pixelsRT.texture;
  u.gsd.value = gsd;
  renderer.setRenderTarget(null);
  canvasDrawPass.render(renderer);
}

export function updateShaderDefines(params, name = null) {
  if (nextSplatPass && name != 'shadows') {
    nextSplatPass.material.updateDefines(params);
  }

  if (raymarchingPass) {
    raymarchingPass.material.updateDefines(params);
  }
}
