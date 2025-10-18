import * as THREE from 'three';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import Stats from 'stats.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import {
  MeshBVHHelper, MeshBVH,
  computeBoundsTree, disposeBoundsTree,
  SAH, CENTER, AVERAGE,
  BVHShaderGLSL,
  MeshBVHUniformStruct,
  FloatVertexAttributeTexture
} from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

const params = {
  open: () => loadSceneFile(),
  //size: () => [28 * 300 / window.devicePixelRatio, 40 * 300 / window.devicePixelRatio],
  size: () => [window.innerWidth, window.innerHeight],

  mode: 'points',
  render: true,
  strategy: SAH,
  maxDepth: 40,
  maxLeafTris: 8,
  sparsity: 0,
  invertY: false,
  maxStdDev: 1.5, // exp2, same as maxStdDev in https://sparkjs.dev
  splatScale: 0, // exp2
  splatOpacity: 0, // exp2, density that absorbs light 
  brightness: 0, // exp2, brightness of sunlight or of the splats themselves
  ambientLight: -6, // exp2
  rayStep: -2.0, // exp10
  fogDensity: -10, // exp2
  shadows: false,
  monochrome: false,
  lightPos: new THREE.Vector3(1e3, 2e3, 3e3),
  shadowMapLayers: 1,
};

const getBVHOptions = () => ({
  strategy: params.strategy,
  maxDepth: params.maxDepth,
  maxLeafTris: params.maxLeafTris,
});

let renderer, camera, scene, orbit, gui, stats, outputContainer;
let bvh, bvhGeometry, bvhHelper, pointCloud;
let raytracingPass, nextSplatPass, raymarchingPass, shadowMapPass, outputPass;
let pixelsRT1, pixelsRT2;
let splatColorsRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
let shadowMapRT = new THREE.WebGLArrayRenderTarget(1, 1, 1, { type: THREE.UnsignedShortType, format: THREE.RedFormat });
let frameId = 0;

//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';
const sceneFile = 'models/soundform.ply';

class GSplatsDataUniformStruct {
  splatsCount = bvhGeometry.attributes.position.count; // (xyz, radius) x N
  maxStdDev = 2 ** params.maxStdDev;
  splatOpacity = 2 ** params.splatOpacity;
  brightness = 2 ** params.brightness;
  ambientLight = 2 ** params.ambientLight;
  fogDensity = params.fogDensity > -10 ? 2 ** params.fogDensity : 0;
  monochrome = params.monochrome;
  shadowMap = shadowMapRT.texture;
  splatColors = splatColorsRT.texture;
}

THREE.ShaderChunk['yuv_rgb'] = /* glsl */`
  const mat3 YUV_RGB = transpose(mat3(1,1,1,  0,-0.34,1.77, 1.4,-0.72,0));
  const mat3 RGB_YUV = inverse(YUV_RGB); // yuv = rgb * RGB_YUV

  float vmax3(vec3 v) { return max(max(v.x, v.y), v.z); }
  float vmin3(vec3 v) { return -vmax3(-v); }
  float vmid3(vec3 v) { return (vmax3(v) + vmin3(v))*0.5; }
`;

THREE.ShaderChunk['gsplats_data'] = /* glsl */`
  #define USE_GAMMA 1 // blend RGB^2, then output sqrt(RGB)

  struct GSplatsData {
    int splatsCount;
    float maxStdDev;
    float splatOpacity;
    float brightness;
    float ambientLight;
    float fogDensity;
    bool monochrome;
    sampler2DArray shadowMap;
    sampler2D splatColors;
  };
`;

THREE.ShaderChunk['unpack_4x16'] = /* glsl */`
  // float 0..1 <-> uint 0..65535
  #define PACK_2x16(xy)   uintBitsToFloat(packUnorm2x16(xy))
  #define PACK_4x16(v)    vec2(PACK_2x16(v.xy), PACK_2x16(v.zw))
  #define UNPACK_2x16(x)  unpackUnorm2x16(floatBitsToUint(x))
  #define UNPACK_4x16(v)  vec4(UNPACK_2x16(v.x), UNPACK_2x16(v.y))
`;

THREE.ShaderChunk['pack_pixel_data'] = /* glsl */`
  struct PixelData { vec4 color; float zDepth; int cost; };

  vec4 packPixelData(PixelData pd) {
    vec2 xy = PACK_4x16(pd.color);
    vec2 zw = vec2(pd.zDepth, pd.cost);
    return vec4(xy, zw);
  }

  PixelData unpackPixelData(vec4 pixel) {
    PixelData pd;
    pd.color = UNPACK_4x16(pixel.xy);
    pd.zDepth = pixel.z;
    pd.cost = int(pixel.w);
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
    if (len < 0.001)
      return len*gaussian3d(pos);

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

    if (abs(len * d) < 0.001)
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

  void bvhInitBounds() {
    vec3 aa = texelFetch1D( bvh.bvhBounds, 0u ).xyz;
    vec3 bb = texelFetch1D( bvh.bvhBounds, 1u ).xyz;
    bvhBounds = mat2x3(aa, bb);
  }

  vec4 integrateSplat(vec4 splat, vec4 color, vec3 pos, vec3 dir, float len) {
    if (splat.w < 1e-6)
      return vec4(0);

    #if NEED_COLOR
      if (gsd.monochrome)
        color.rgb = vec3(1)*vmid3(color.rgb);

      #if USE_GAMMA
        color.rgb *= color.rgb;
      #endif
    #endif
    
    color.w *= gsd.splatOpacity;
    
    #if NEED_COLOR
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

  void bvhInitSearch() {
    gSumShadow = 0.;
    gSumColor = vec4(0);

    #if NEED_SHADOW
      // find the nearest shadow map layer towards the sun
      vec3 aa = bvhBounds[0], bb = bvhBounds[1];
      vec3 p = (gPos - aa) / (bb - aa); // 0..1 x 0..1 x -INF..1
      float numLayers = float(textureSize(gsd.shadowMap, 0).z);
      float layer = max(ceil(numLayers * p.z), float(gMinShadowMapLayer));
      float delta = layer/numLayers - p.z; // 0..1
      vec3 sun = gSunDir / (bb - aa); // gSunDir.z = 1.0 + eps
      vec2 uv = p.xy + sun.xy/sun.z * delta; // uv.z = layer/numLayers

      if (layer < numLayers) {
        gSumShadow += 8.0*texture(gsd.shadowMap, vec3(uv, layer)).x;
        bvhTexLookups++;
        gSunDist = delta * (bb.z - aa.z)/gSunDir.z;
        gSunPos = gPos + gSunDir*gSunDist;
      }
    #endif

    #if NEED_COLOR
      gSumColor = integrateSplat(gFogSplat, gFogColor, gPos, gRayDir, 1e-9)/1e-9*RAY_STEP;
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
        vec4 dens = integrateSplat(splat, color, gPos, gRayDir, 1e-9)/1e-9*RAY_STEP;
        gSumColor += dens;
      }
    #endif

    return color.w > 0.;
  }
`;

// Find the next splat starting from rayOrigin + rayDir*abs(pixelData.z).
// It's a no-op if pixelData.z > 0. because there is a splat at that point. 
class NextSplatMaterial extends THREE.ShaderMaterial {

  updateDefines() {
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
            rayData.w = 0.;
          }

          if (rayData.z < 0. && abs(rayData.z) < INFINITY) {
            bvhRayDir = rayDir;
            bvhRay.origin = rayOrigin + rayDir * abs(rayData.z);
            
            bvhSearchSplats( bvh );
            
            rayData.z = abs(rayData.z) + bvhNearest.dist + RAY_STEP*0.5;
            rayData.w += float(bvhTexLookups);
          }

          gl_FragColor = rayData;
        }`
    });
  }
}

// Computes density and shadows at the current point and makes a RAY_STEP forward.
// It's a no-op if pixelData.z < 0. because there are no splats at the current point.
class RaymarchingMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.RAY_STEP = 10 ** params.rayStep;
    this.defines.NEED_SHADOW = +params.shadows;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      defines: {

        RAY_STEP: 0.001,
        BVH_STACK_DEPTH: 64,
        INTEGRATE_FOG: 0,
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

        // Fast fog integration outside the AABB bounding box.
        vec4 integrateFog(vec3 pos, vec3 dir, float len) {
          const int N = 64;
          vec4 sum = vec4(0);

          for (int i = 0; i < N; i++) {
            float dt = len / float(N);
            float t = float(i)*dt;
            vec3 p = pos + dir*t;
            vec3 sunDir = gSunPos - p;
            float sunDist = length(sunDir);
            sunDir /= sunDist;

            float lum = gsd.brightness;

            #if NEED_SHADOW
              float weight = integrateSplat(gFogSplat, gFogColor, p, sunDir, sunDist).w;
              lum *= exp(-weight);
              lum += gsd.ambientLight; // ambient occlusion (AO) or global illumination (GI)
            #endif

            vec4 vol = integrateSplat(gFogSplat, gFogColor, p, dir, 1e-9)/1e-9*dt;
            vol.rgb *= lum;
            
            sum.rgb += exp(-sum.w) * vol.rgb;
            sum.w += vol.w;
          }

          return sum;
        }

        // Integrates fog outside the AABB and advances the ray to the AABB.
        bool skipFog(vec3 rayOrigin, vec3 rayDir, inout vec4 rayData, inout vec4 color) {
          vec3 aa = bvhBounds[0];
          vec3 bb = bvhBounds[1];

          aa.z -= (bb.z - aa.z)*15.; // the shadow that the AABB box casts

          // rayOrigin doesn't change, but rayData.z does
          vec2 tt = rayBox(rayOrigin, rayDir, aa, bb);
          tt.x = max(tt.x, 0.);
          
          if (tt.x >= tt.y)
            tt = vec2(INFINITY);

          float maxFogDist = length(aa - bb);

          if (frameId == 0) {
            rayData.z = tt.x + RAY_STEP*0.5;
            
            if (rayData.z > RAY_STEP) {
              #if INTEGRATE_FOG
                // add fog that's in front of the AABB
                float len = min(rayData.z, maxFogDist);
                vec3 pos = rayOrigin;
                if (rayData.z < INFINITY)
                  pos += rayDir*(rayData.z - len);
                vec4 fog = integrateFog(pos, rayDir, len);
                color.rgb += exp(-color.w) * fog.rgb;
                color.w += fog.w;
              #endif

              return true;
            }
          } else if (rayData.z > tt.y) {
            #if INTEGRATE_FOG
              // add fog that's behind the AABB
              vec4 fog = integrateFog(rayOrigin + rayData.z*rayDir, rayDir, maxFogDist);
              color.rgb += exp(-color.w) * fog.rgb;
              color.w += fog.w;
            #endif

            rayData.z = INFINITY;
            return true;
          }

          return false;
        }

        bool blendSplats(vec3 rayOrigin, vec3 rayDir, inout vec4 rayData, inout vec4 color) {
          bvhSearchSplats( bvh );
          rayData.w += float(bvhTexLookups);
          
          if (isnan(dot(gSumColor,gSumColor)) || isnan(gSumShadow))
            gSumColor = vec4(0), gSumShadow = 0.;

          if (gSumColor.w <= 0. && gFogSplat.w == 0.)
            return false;

          float luminance = gsd.brightness;

          #if NEED_SHADOW
            gSumShadow += integrateSplat(gFogSplat, gFogColor, gPos, gSunDir, INFINITY).w;
            luminance *= exp(-gSumShadow);
            luminance += gsd.ambientLight; // ambient occlusion (AO) or global illumination (GI)
          #endif

          vec4 vol = gSumColor;
          vol.rgb *= luminance;
          
          color.rgb += exp(-color.w) * vol.rgb;
          color.w += vol.w;

          rayData.z += RAY_STEP;
          return true;
        }

        vec4 compress(vec4 rayData, vec4 color) {
          color.w /= 8.;
          if (color.w > 0.999)
              rayData.z = INFINITY;
          rayData.xy = PACK_4x16(color);
          return rayData;
        }

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDir;
          ndcToCameraRay(
            ndc, inverse(modelWorldMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDir);
          rayDir = normalize(rayDir);

          // .xy = accumulated color + density, packed as 4 x float16
          // .z = current Z depth for raycasting
          // .w = accumulated cost
          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);
          bool hasFog = gsd.fogDensity > 0.;

          if (frameId == 0) {
            rayData.xy = PACK_4x16(vec4(0));
            // NextSplatMaterial runs first when fog=0
            if (hasFog) rayData.zw = vec2(0);
          }

          if (rayData.z < 0.)
            discard; // it's NextSplatMaterial's turn

          // skip already rendered pixels
          if (rayData.z >= INFINITY) {
            gl_FragColor = rayData;
            return;
          }

          if (hasFog) {
            // (aa, bb) is in sun coords, so vec3(0,0,1) points to the sun
            gFogSplat = vec4(0, 0, -3, 3);
            gFogColor = vec4(1, 1, 1, gsd.fogDensity/gsd.splatOpacity);
          }

          gRayDir = rayDir;
          gPos = rayOrigin + rayDir * rayData.z;
          gSunPos = (vec4(lightPos, 1) * inverse(modelWorldMatrix)).xyz;
          gSunDir = normalize(gSunPos - gPos);
          gSunDist = length(gSunPos - gPos);
          bvhRayDir = gSunDir;

          bvhInitBounds();

          vec4 color = UNPACK_4x16(rayData.xy);
          color.w *= 8.; // density range: exp(0)..exp(-8) = 1..0.0003

          if (hasFog) {
            if (skipFog(rayOrigin, rayDir, rayData, color)) {
              gl_FragColor = compress(rayData, color);
              return;
            }
          }

          if (!blendSplats(rayOrigin, rayDir, rayData, color))
            rayData.z *= -1.; // let NextSplatMaterial find the next splat
          
          gl_FragColor = compress(rayData, color);
        }`
    });
  }
}

class ShadowMapMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.needsUpdate = true;
  }

  constructor() {

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
          bool hasFog = gsd.fogDensity > 0.;
          int numLayers = textureSize(gsd.shadowMap, 0).z;

          if (hasFog) {
            // (aa, bb) is in sun coords, so vec3(0,0,1) points to the sun
            gFogSplat = vec4(0, 0, -3, 3);
            gFogColor = vec4(1, 1, 1, gsd.fogDensity/gsd.splatOpacity);
          }

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

// Finds the nearest 8 splats, blends them, then repeats the same at the next frame.
// In practice, it's better to use a proper rasterizer: https://sparkjs.dev.
class RaytracingMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      uniforms: {

        frameId: { value: 0 },
        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },
        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelWorldMatrix: { value: new THREE.Matrix4() },

      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1 );
        }
      `,

      fragmentShader: /* glsl */`
        in vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}

        #include <ray_utils>
        #include <gsplats_data>
        #include <common>
        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <pack_pixel_data>
        #include <gaussian_utils>

        uniform BVH bvh;
        uniform GSplatsData gsd;
        uniform sampler2D pixelData;
        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelWorldMatrix;
        uniform int frameId;

        ///// BVH traversal ///////////////////////////////////////////////////

        vec3 bvhRayDir;
        vec3 gRayOrigin;
        mat4 gSplats; // the closest 8 splats sorted by distance

        void bvhInitSearch() {
          vec4 s = vec4(INFINITY);
          gSplats = mat4(s, s, s, s);
        }

        bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
          vec2 tt = rayBox( gRayOrigin, bvhRayDir, boundsMin, boundsMax );
          return tt.x < tt.y && tt.x < gSplats[3].z;
        }

        bool bvhVisitSplat(uint splatId) {
          vec4 splat = texelFetch1D( bvh.position, splatId );
          
          if (splat.w < 1e-6)
              return false;
          
          vec3 r = (gRayOrigin - splat.xyz) / splat.w;
          float t = dot(r, -bvhRayDir);
          float h = t*t + 1. - dot(r, r);

          if (h <= 0. || t <= 0. || t*splat.w >= gSplats[3].z)
            return false;          

          // this list is sorted by .x:
          //    a.xy, a.zw, b.xy, b.zw, 
          //    c.xy, c.zw, d.xy, d.zw ... s.xy
          vec4 a = gSplats[0];
          vec4 b = gSplats[1];
          vec4 c = gSplats[2];
          vec4 d = gSplats[3];
          vec2 s = vec2(t*splat.w, splatId);
          
          if (s.x < d.z) d.zw = s.xy;
          if (d.z < d.x) d = d.zwxy;
          if (d.x < c.z) s = c.zw, c.zw = d.xy, d.xy = s; // swap(c.zw, d.xy)
          if (c.z < c.x) c = c.zwxy;
          if (c.x < b.z) s = b.zw, b.zw = c.xy, c.xy = s; // swap(b.zw, c.xy)
          if (b.z < b.x) b = b.zwxy;
          if (b.x < a.z) s = a.zw, a.zw = b.xy, b.xy = s; // swap(a.zw, b.xy)
          if (a.z < a.x) a = a.zwxy;

          gSplats = mat4(a, b, c, d);
          return true;
        }
        
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}

        ///// GSplat color blending ///////////////////////////////////////////
        
        void blendSplat(vec2 entry, inout vec4 sumColor) {
          uint splatId = uint(entry.y);
          float dist = entry.x;

          if (dist >= INFINITY || sumColor.w >= 1.0)
            return;
          
          vec4 splat = texelFetch1D(bvh.position, splatId);
          vec4 color = texelFetch1D(gsd.splatColors, splatId);

          if (gsd.monochrome)
            color.rgb = vec3(1)*vmid3(color.rgb);

          #if USE_GAMMA
            color.rgb *= color.rgb; // blend RGB^2, then output sqrt(RGB)
          #endif

          color.rgb *= gsd.brightness;
          color.w *= gsd.splatOpacity;

          // rasterizer-style blending: splats are approximated with flat ellipses
          vec3 r = (gRayOrigin + bvhRayDir*dist - splat.xyz) / splat.w;
          color.w *= gaussian3d(r * gsd.maxStdDev / SQRT_2);

          color.rgb *= color.w;
          sumColor += (1. - sumColor.w) * color;
        }
        
        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDirection;
          ndcToCameraRay(
            ndc, inverse(modelWorldMatrix) * cameraWorldMatrix,
            inverse(projectionMatrix),
            rayOrigin, rayDirection);

          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 pixel = texelFetch(pixelData, ivec2(vUv*size), 0);
          PixelData pd = unpackPixelData(pixel);

          if (frameId > 0 && pd.zDepth >= INFINITY) {
            gl_FragColor = pixel;
            return;
          }

          if (frameId == 0) {
            pd.color = vec4(0);
            pd.zDepth = 0.;
            pd.cost = 0;
          }
          
          bvhRayDir = normalize(rayDirection);
          gRayOrigin = rayOrigin + pd.zDepth * bvhRayDir;
          bvhSearchSplats( bvh );
          pd.cost += bvhTexLookups; // total cost
          vec4 rgba = pd.color;

          blendSplat(gSplats[0].xy, rgba);
          blendSplat(gSplats[0].zw, rgba);
          blendSplat(gSplats[1].xy, rgba);
          blendSplat(gSplats[1].zw, rgba);
          blendSplat(gSplats[2].xy, rgba);
          blendSplat(gSplats[2].zw, rgba);
          blendSplat(gSplats[3].xy, rgba);
          blendSplat(gSplats[3].zw, rgba);
          
          pd.color = rgba;
          pd.zDepth += (1. + 1e-6) * gSplats[3].z + 1e-6;
          gl_FragColor = packPixelData(pd);
        }`
    });
  }
}

class CanvasDrawMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
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

        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <gsplats_data>
        #include <pack_pixel_data>

        ${BVHShaderGLSL.common_functions}

        uniform GSplatsData gsd;
        vec2 size;

        void drawShadowMap(inout vec4 o) {
          vec2 uv = vUv*5.0 - vec2(0,4);
          if (uv != clamp(uv, 0., 1.))
            return;

          int numLayers = textureSize(gsd.shadowMap, 0).z;
          int layer = 0; // frameId/120 % numLayers;
          float shadow = 8.0*texture(gsd.shadowMap, vec3(uv, layer)).x;
          o.rgb = vec3(1,3,9) * (1. - exp(-shadow));
        }

        void drawProgress(inout vec4 o) {
          vec2 uv = vUv*5.0 - vec2(0,0);
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
          vec2 uv = vUv*5.0 - vec2(4,0);
          if (uv != clamp(uv, 0., 1.))
            return;

          vec4 pixel = texture(pixelData, uv);
          PixelData pd = unpackPixelData(pixel);
          o.rgb = vec3(9,3,1) * float(pd.cost)/1e5;
        }

        void main() {
          size = vec2(textureSize(pixelData, 0));
          vec4 pixel = texelFetch(pixelData, ivec2(vUv*size), 0);
          PixelData pd = unpackPixelData(pixel);
          vec4 o = pd.color;

          #if USE_GAMMA
            o.rgb = sqrt(o.rgb); // gamma correction
          #endif
          //o.rgb += exp(-o.w) * background;

          //drawShadowMap(o);
          drawProgress(o);
          drawCost(o);

          o.w = 1.0;

          if (isnan(dot(o,o)))
            o.rgb = vec3(0,1,0);

          gl_FragColor = o;
        }`
    });
  }
}

// (f_dc, rgb, opacity) -> rgba
class SplatColorsMaterial extends THREE.ShaderMaterial {
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

async function init() {
  outputContainer = document.getElementById('output');

  // renderer setup
  let [w, h] = params.size();
  renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h, false);
  renderer.setClearColor(0, 0);
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();

  let light = new THREE.DirectionalLight(0xFFFFFF);
  light.position.set(params.lightPos);
  scene.add(light);

  camera = new THREE.PerspectiveCamera(60, w / h, 0.001, 50);
  camera.position.set(1, 1, 2);
  camera.far = 100;
  camera.updateProjectionMatrix();

  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.addEventListener('change', () => clearRenderTargets());
  orbit.addEventListener('start', () => { orbit.interacting = true; });
  orbit.addEventListener('end', () => { orbit.interacting = false; });

  pixelsRT1 = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });
  pixelsRT2 = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });

  stats = new Stats();
  document.body.appendChild(stats.dom);

  outputPass = new FullScreenQuad(new CanvasDrawMaterial());
  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();
  nextSplatPass = new FullScreenQuad(new NextSplatMaterial());
  nextSplatPass.material.updateDefines();
  raymarchingPass = new FullScreenQuad(new RaymarchingMaterial());
  raymarchingPass.material.updateDefines();
  shadowMapPass = new FullScreenQuad(new ShadowMapMaterial());
  shadowMapPass.material.updateDefines();

  window.params = params;
  window.THREE = THREE;
  window.renderer = renderer;

  initGeometry();
  rebuildGUI();

  updateRenderSize();
  window.addEventListener('resize',
    () => updateRenderSize(), false);
}

function updateRenderSize() {
  let [w, h] = params.size();
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  renderer.setSize(w, h, false);
  clearRenderTargets();
}

function clearRenderTargets() {
  let [w, h] = params.size();
  pixelsRT1.setSize(w, h);
  pixelsRT2.setSize(w, h);
  frameId = 0;
}

async function loadSceneFile() {
  let input = document.createElement('input');
  input.type = 'file';
  input.accept = '*.ply';
  input.multiple = false;
  input.click();
  let blob = await new Promise(resolve =>
    input.onchange = () => resolve(input.files[0]));
  if (!blob) return;

  frameId = -1;

  console.log('Opening file:', (blob.size / 1e6).toFixed(1), 'MB', blob.name);
  let url = URL.createObjectURL(blob);
  await initGeometry(url, blob.name);
  URL.revokeObjectURL(url);

  clearRenderTargets();
}

async function loadGeometry(url, filename = url) {
  console.time('loadGeometry');
  console.log('Loading scene:', filename);

  let geometry = filename.endsWith('.ply') ?
    await loadPLY(url) :
    await loadGLTF(url);

  //geometry.center();
  console.timeEnd('loadGeometry');
  return geometry;
}

async function loadPLY(url) {
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

async function loadGLTF(url) {
  let gltf = await new GLTFLoader()
    .setMeshoptDecoder(MeshoptDecoder)
    .loadAsync(url);
  gltf.scene.updateMatrixWorld(true);
  let gltfMesh = gltf.scene.children[0];
  return gltfMesh.geometry;
}

async function initGeometry(url = sceneFile, filename) {
  const geometry = await loadGeometry(url, filename);
  const material = new THREE.PointsMaterial({ color: 0xCCCCCC });
  scene.remove(pointCloud);
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  let sunMatrix = getSunMatrix4(params.lightPos);
  console.debug('det(sunMatrix) = ' + sunMatrix.determinant());
  pointCloud.geometry.applyMatrix4(sunMatrix.clone().invert());
  pointCloud.matrix = sunMatrix;
  pointCloud.matrixAutoUpdate = false;
  pointCloud.updateMatrixWorld();

  updateBVHMesh();
  updateSplatColors();
  updateShadowMap();
}

function getSunMatrix4(zAxis) {
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

function updateBVHMesh() {
  bvhGeometry = new THREE.BufferGeometry();
  let attributes = pointCloud.geometry.attributes;

  let m = 1 << params.sparsity;
  let position = attributes.position;
  let numSplats = position.count;
  let numSplatsM = numSplats / m | 0;

  if (numSplatsM > 1e5) console.time('updateBVH');
  let str = numSplatsM < 1e3 ? numSplatsM : (numSplatsM / 1e3).toFixed(0) + 'K';
  outputContainer.textContent = str + ' splats';

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

  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position3);
  bvhGeometry.computeBoundsTree(getBVHOptions());

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

  // GLSL needs 4-element position attr for efficiency, but MeshBVH doesn't support that,
  // so build the BVH first, and then replace the position attr, as MeshBVH no longer needs it.
  bvh = new MeshBVH(bvhGeometry, getBVHOptions());
  bvhGeometry.attributes.position.copy(position4);
  position3 = null; // it's been replaced with position4

  let bbox = new THREE.Box3();
  bvh.getBoundingBox(bbox);
  let dx = bbox.max.x - bbox.min.x;
  let dy = bbox.max.y - bbox.min.y;
  let dz = bbox.max.z - bbox.min.z;
  console.debug('Bounding box:', dx.toFixed(2), 'x', dy.toFixed(2), 'x', dz.toFixed(2));

  if (numSplatsM > 1e5) console.timeEnd('updateBVH');
}

function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();
  gui.onChange((e) => {
    if (e.property != 'render')
      clearRenderTargets();
  });

  gui.add(params, 'open');

  gui.add(params, 'render').onChange(() => {
    orbit.enabled = params.render;
  });

  const pointsFolder = gui.addFolder('points');

  pointsFolder.add(params, 'maxDepth', 4, 64, 1).onChange(v => {
    nextSplatPass.material.updateDefines();
    raymarchingPass.material.updateDefines();
    shadowMapPass.material.updateDefines();
    updateBVHMesh();
    updateShadowMap();
  });
  pointsFolder.add(params, 'sparsity', 0, 16, 1).onChange(v => {
    updateBVHMesh();
    updateShadowMap();
  });
  pointsFolder.add(params, 'invertY').onChange(v => {
    pointCloud.matrix.scale(new THREE.Vector3(1, 1, -1));
    bvhHelper.mesh.matrix = pointCloud.matrix.clone();
    clearRenderTargets();
    updateShadowMap();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'raytracing', 'raymarching']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raymarching') {
    displayFolder.add(params, 'rayStep', -4, -1, 0.5).onChange(() => {
      nextSplatPass.material.updateDefines();
      raymarchingPass.material.updateDefines();
    });
    displayFolder.add(params, 'fogDensity', -10, +10, 0.5).onChange(() => {
      updateShadowMap();
    });
    displayFolder.add(params, 'ambientLight', -10, -1, 0.5);
    displayFolder.add(params, 'shadows').onChange(() => {
      raymarchingPass.material.updateDefines();
      shadowMapPass.material.updateDefines();
      updateShadowMap();
    });
    displayFolder.add(params, 'shadowMapLayers', 1, 32, 1).onChange(() => {
      updateShadowMap();
    });
  }

  if (params.mode == 'raytracing' || params.mode == 'raymarching') {
    displayFolder.add(params, 'maxStdDev', 0, 3, 0.5).onChange(() => {
      updateBVHMesh();
      updateShadowMap();
    });
    displayFolder.add(params, 'splatScale', -4, 4, 0.5).onChange(() => {
      updateBVHMesh();
      updateShadowMap();
    });
    displayFolder.add(params, 'splatOpacity', -4, 8, 0.5).onChange(() => {
      updateShadowMap();
    });
    displayFolder.add(params, 'brightness', -3, 3, 0.5);
    displayFolder.add(params, 'monochrome');
  }
}

function updateSplatColors() {
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
}

function updateShadowMap() {
  if (!params.shadows)
    return;

  console.time('Update shadowMap');
  pointCloud.updateMatrixWorld();

  let size = 2048, layers = params.shadowMapLayers;
  shadowMapRT.setSize(size, size, layers);

  let layerRT = new THREE.WebGLRenderTarget(size, size, { type: THREE.FloatType, format: THREE.RedFormat });
  let copy = new FullScreenQuad(new THREE.ShaderMaterial(CopyShader));
  copy.material.uniforms.tDiffuse.value = layerRT.texture;

  let u = shadowMapPass.material.uniforms;
  u.bvh.value.updateFrom(bvh);
  u.gsd.value = new GSplatsDataUniformStruct();
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

  layerRT.dispose();
  copy.dispose();

  // Read data to CPU to measure time correctly.
  // This wouldn't work with a 3D texture.
  renderer.readRenderTargetPixels(shadowMapRT,
    0, 0, 1, 1, new Uint8Array(4), 0);
  console.timeEnd('Update shadowMap');
  clearRenderTargets();
}

function render() {

  requestAnimationFrame(render);

  if (frameId < 0 || !params.render)
    return;

  stats.update();

  if (params.mode === 'points' || orbit.interacting) {

    if (!pointCloud) return;
    pointCloud.material.size = 0.005;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

  } else if (params.mode === 'raytracing' || params.mode == 'raymarching') {
    if (!bvh) return;

    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    let uniforms;
    let gsd = new GSplatsDataUniformStruct();

    if (params.mode == 'raytracing') {
      uniforms = raytracingPass.material.uniforms;
      uniforms.bvh.value.updateFrom(bvh);
      uniforms.frameId.value = frameId;
      uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
      uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
      uniforms.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
      uniforms.pixelData.value = pixelsRT1.texture;
      uniforms.gsd.value = gsd;
      renderer.setRenderTarget(pixelsRT2);
      raytracingPass.render(renderer);
    }

    if (params.mode == 'raymarching') {
      if (!gsd.fogDensity) {
        uniforms = nextSplatPass.material.uniforms;
        uniforms.bvh.value.updateFrom(bvh);
        uniforms.gsd.value = gsd;
        uniforms.pixelData.value = pixelsRT1.texture;
        uniforms.frameId.value = frameId;
        uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
        uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
        uniforms.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
        renderer.setRenderTarget(pixelsRT2);
        nextSplatPass.render(renderer);

        [pixelsRT1, pixelsRT2] = [pixelsRT2, pixelsRT1];
      }

      uniforms = raymarchingPass.material.uniforms;
      uniforms.bvh.value.updateFrom(bvh);
      uniforms.gsd.value = gsd;
      uniforms.pixelData.value = pixelsRT1.texture;
      uniforms.frameId.value = frameId;
      uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
      uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
      uniforms.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
      renderer.setRenderTarget(pixelsRT2);
      raymarchingPass.render(renderer);
    }

    uniforms = outputPass.material.uniforms;
    uniforms.frameId.value = frameId;
    uniforms.pixelData.value = pixelsRT2.texture;
    uniforms.gsd.value = gsd;
    renderer.setRenderTarget(null);
    outputPass.render(renderer);

    [pixelsRT1, pixelsRT2] = [pixelsRT2, pixelsRT1];
    frameId++;
  }
}

init();
render();
