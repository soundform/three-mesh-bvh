import * as THREE from 'three';
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
  open: () => selectScene(),

  mode: 'points',
  render: true,
  strategy: SAH,
  maxDepth: 32,
  maxLeafTris: 8,
  sparsity: 0,
  splatScale: 1.5, // exp2, same as maxStdDev in https://sparkjs.dev
  splatOpacity: 0, // exp2, density that absorbs light 
  brightness: 0, // exp2, brightness of sunlight or of the splats themselves
  maxSplatsPerRay: 8,
  rayStep: -2.5, // 10**-2.5
  shadows: false,
  showCost: false,
  showProgress: false,
};

const getBVHOptions = () => ({
  strategy: params.strategy,
  maxDepth: params.maxDepth,
  maxLeafTris: params.maxLeafTris,
});

let renderer, camera, scene, orbit, gui, stats, outputContainer;
let bvh, bvhMesh, bvhHelper, pointCloud;
let raytracingPass, nextSplatPass, raymarchingPass, outputPass;
let pixelsRT1, pixelsRT2, splatColorsRT;
let dummyRT = new THREE.WebGLRenderTarget();
let lightDir = new THREE.Vector3(-1, 1, 1).normalize();
let frameId = 0;

//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';
//const sceneFile = 'models/bunny.glb';
const sceneFile = 'models/soundform.ply';
//const sceneFile = 'models/sportcar.ply';

class GSplatsDataUniformStruct {
  splatsCount = pointCloud.geometry.attributes.position.count;
  splatScale = 2 ** params.splatScale;
  splatOpacity = 2 ** params.splatOpacity;
  brightness = 2 ** params.brightness;
  splatColors = splatColorsRT.texture;
}

THREE.ShaderChunk['yuv_rgb'] = /* glsl */`
  const mat3 YUV_RGB = mat3(1,1,1,  0,-0.34,1.77, 1.4,-0.72,0);
  const mat3 RGB_YUV = inverse(YUV_RGB); // yuv = RGB_YUV * rgb
`;

THREE.ShaderChunk['gsplats_data'] = /* glsl */`
  #define USE_GAMMA 1 // blend RGB^2, then output sqrt(RGB)

  struct GSplatsData {
    int splatsCount;
    
    float splatScale;
    float splatOpacity;
    float brightness;
    
    sampler2D splatColors;
  };
`;

THREE.ShaderChunk['unpack_4x16'] = /* glsl */`
  #define PACK_2x16(xy)   uintBitsToFloat(packHalf2x16(xy))
  #define PACK_4x16(v)    vec2(PACK_2x16(v.xy), PACK_2x16(v.zw))
  #define UNPACK_2x16(x)  unpackHalf2x16(floatBitsToUint(x))
  #define UNPACK_4x16(v)  vec4(UNPACK_2x16(v.x), UNPACK_2x16(v.y))
`;

THREE.ShaderChunk['gaussian_utils'] = /* glsl */`
  const float SQRT_PI = sqrt(radians(180.));
  const float SQRT_2 = sqrt(2.0);

  float gaussian3d(vec3 r) {
    return exp(-dot(r,r));
  }
  
  // integrate(exp(-x*x))*2/sqrt(PI)
  // https://en.wikipedia.org/wiki/Error_function
  float erfc(float x) {
    if (x < -3.5) return -1.; // optional
    if (x > +3.5) return +1.; // optional
    return sign(x)*sqrt(1. - exp2(-SQRT_PI*x*x)); // -1..1
  }

  // integrate(exp(-|pos + dir*t|^2))*2/sqrt(PI)
  // https://en.wikipedia.org/wiki/Gaussian_integral
  float erfc_3d(vec3 pos, vec3 dir, float tmin, float tmax) {
    float b = dot(pos, dir);                    // -INF..INF
    float h = dot(pos, pos) - b*b;              // 0..INF
    float s = erfc(b + tmax) - erfc(b + tmin);  // 0..2
    return exp(-h)*s*SQRT_PI*0.5;               // 0..sqrt(PI)
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

// Uses BVH to find the nearest N splats along the ray. 
THREE.ShaderChunk['bvh_sorted_splats'] = /* glsl */`
  #ifndef MAX_SPLATS_PER_RAY
  #define MAX_SPLATS_PER_RAY 1
  #endif

  struct BVHRay { vec3 origin, dir; } bvhRay;

  // The nearest N splats are sorted by distance.
  // It's possible to use a min-heap instead, but:
  //  1) GPUs don't like the min-heap read/write patterns.
  //  2) The splats need to be sorted anyway for color blending.
  float[MAX_SPLATS_PER_RAY] gSplatDists;
  uint[MAX_SPLATS_PER_RAY] gSplatIds;
  int gNumSplats; // 0..MAX_SPLATS_PER_RAY

  void bvhInitSearch() {
    for (int i = 0; i < MAX_SPLATS_PER_RAY; i++)
      gSplatDists[i] = INFINITY;
    gNumSplats = 0;
  }

  bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
    vec2 tt = rayBox( bvhRay.origin, bvhRay.dir, boundsMin, boundsMax );
    return tt.x < tt.y && tt.x < gSplatDists[MAX_SPLATS_PER_RAY-1];
  }

  bool bvhVisitSplat(uint splatId) {
    vec4 splat = texelFetch1D( bvh.position, splatId );
    vec2 tt = raySphere(bvhRay.origin - splat.xyz, bvhRay.dir, splat.w);
    float d = (tt.x + tt.y)*0.5;
    
    if (tt.x >= tt.y || d <= 0. || d >= gSplatDists[MAX_SPLATS_PER_RAY-1])
      return false;

    // insert the new sample point into the sorted list
    gNumSplats = min(gNumSplats + 1, MAX_SPLATS_PER_RAY);

    for (int k = gNumSplats - 1; k >= 0; k--) {
      if (d >= gSplatDists[k])
        break;

      if (k + 1 < MAX_SPLATS_PER_RAY) {
        gSplatDists[k + 1] = gSplatDists[k];
        gSplatIds[k + 1] = gSplatIds[k];
      }

      gSplatDists[k] = d;
      gSplatIds[k] = splatId;
    }

    return true;
  }
`;

// Uses BVH to compute aggregate density and shadow at the current spot.
THREE.ShaderChunk['bvh_shadows_raycasting'] = /* glsl */`
  struct BVHRay { vec3 origin, dir; } bvhRay;
  const float MAX_TOTAL_DENSITY = -log(0.001);
  float bvhSumShadow;
  vec4 bvhSumColor;

  void bvhInitSearch() {
    bvhSumShadow = 0.;
    bvhSumColor = vec4(0);
  }

  bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
    if (bvhSumShadow > MAX_TOTAL_DENSITY)
      return false;
    
    #if USE_SHADOWS
    
      vec2 tt = rayBox( bvhRay.origin, bvhRay.dir, boundsMin, boundsMax );
      tt = max(tt, vec2(0));
      return tt.x < tt.y && tt.y > 0.;

    #else 

      return bvhRay.origin == clamp( bvhRay.origin, boundsMin, boundsMax );

    #endif
  }

  bool bvhVisitSplat(uint splatId) {
    if (bvhSumShadow > MAX_TOTAL_DENSITY)
      return false;

    vec4 splat = texelFetch1D( bvh.position, splatId );
    // splat.w = original radius * global scale
    vec3 r = (bvhRay.origin - splat.xyz) / splat.w;
    float rd = dot(r, bvhRay.dir);
    float r2 = dot(r, r);
    float h2 = r2 - rd*rd;
    float scale = SQRT_2 / gsd.splatScale;

    // see if bvhRay intersects the splat
    if (h2 < 1.) {

      #if USE_SHADOWS
      
        // The proper integral would be:
        //
        //    erfc_3d((origin - splat.xyz)/splat.w/scale, dir)*splat.w*scale
        //
        // However rasterizers implicitly scale opacity of splats by their size,
        // and hence the splat.w multiplier is omitted here.
        
        float fog = erfc_3d(r/scale, bvhRay.dir, 0., INFINITY)*scale*splat.w;
        vec4 color = texelFetch1D(gsd.splatColors, splatId);
        color.w *= gsd.splatOpacity/splat.w;
        bvhSumShadow += fog * color.w;

      #endif

      // see if bvhRay.origin is inside the splat
      if (r2 < 1.) {
        
        // Rasterizers render small splats with the same opacity as large splats,
        // but if the splats were to be integrated properly, the opacity would have
        // to be scaled by the splat size: integrate(exp(-1/2 * |r/s|^2)) = sqrt(2*PI)*s
        // This means that rasterizers implicitly scale the density of splats and this
        // has to be accounted for here.
        
        float density = gaussian3d(r/scale);
        vec4 color = texelFetch1D(gsd.splatColors, splatId);
        color.w *= gsd.splatOpacity/splat.w;
        
        #if USE_GAMMA
          color.rgb *= color.rgb;
        #endif

        color.rgb *= color.w;
        bvhSumColor += color * density;
      }
    }

    return h2 < 1.;
  }
`;

// Uses BVH to find the next nearest splat along the ray.
THREE.ShaderChunk['bvh_nearest_splat'] = /* glsl */`
  struct BVHRay { vec3 origin, dir; } bvhRay;
  struct BVHNearest { float dist; uint splatId; } bvhNearest;

  void bvhInitSearch() {
    bvhNearest = BVHNearest(INFINITY, 0u);
  }

  bool bvhVisitBoundingBox(vec3 boundsMin, vec3 boundsMax) {
    vec2 tt = rayBox( bvhRay.origin, bvhRay.dir, boundsMin, boundsMax );
    return tt.x < tt.y && tt.x < bvhNearest.dist && tt.y > 0.;
  }

  bool bvhVisitSplat(uint splatId) {
    vec4 splat = texelFetch1D( bvh.position, splatId );
    vec2 tt = raySphere(bvhRay.origin - splat.xyz, bvhRay.dir, splat.w);
    tt = max(tt, vec2(0));

    if (tt.x < tt.y && tt.x < bvhNearest.dist && tt.y > 0.) {
      bvhNearest.dist = tt.x;
      bvhNearest.splatId = splatId;
      return true;
    }
    
    return false;
  }
`;

// Find the next splat starting from rayOrigin + rayDir*abs(pixelData.z).
// It's a no-op if pixelData.z > 0. because there is a splat at that point. 
class NextSplatMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.SHOW_COST = +params.showCost;
    this.defines.RAY_STEP = 10 ** params.rayStep;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      defines: {

        RAY_STEP: 0.001,
        SHOW_COST: 0,
        BVH_STACK_DEPTH: 64,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelMatrix: { value: new THREE.Matrix4() },

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
        uniform mat4 modelMatrix;

        #include <ray_utils>
        #include <gaussian_utils>
        #include <bvh_nearest_splat>
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        #include <common>
        #include <unpack_4x16>

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDir;
          ndcToCameraRay(
            ndc, inverse(modelMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDir);
          rayDir = normalize(rayDir);

          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          if (frameId == 0)
            rayData.z = -1e-6;

          if (rayData.z < 0. && abs(rayData.z) < INFINITY) {
            bvhRay.dir = rayDir;
            bvhRay.origin = rayOrigin + rayDir * abs(rayData.z);
            bvhSearchSplats( bvh );

            rayData.z = abs(rayData.z) + bvhNearest.dist + RAY_STEP*0.5;

            #if SHOW_COST

              vec4 cost = vec4(bvhStats.numLookupsBVH, bvhStats.numLookupsSplats, bvhStats.numSplats, 0);
              if (frameId > 0)
                cost += UNPACK_4x16(rayData.xy);
              rayData.xy = PACK_4x16(cost);

            #endif
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
    this.defines.SHOW_COST = +params.showCost;
    this.defines.RAY_STEP = 10 ** params.rayStep;
    this.defines.USE_SHADOWS = +params.shadows;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      defines: {

        RAY_STEP: 0.001,
        SHOW_COST: 0,
        BVH_STACK_DEPTH: 64,
        USE_SHADOWS: 1,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelMatrix: { value: new THREE.Matrix4() },

        lightDir: { value: lightDir },
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
        uniform mat4 modelMatrix;
        uniform vec3 lightDir;

        #include <ray_utils>
        #include <gaussian_utils>
        #include <bvh_shadows_raycasting>
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        #include <common>
        #include <yuv_rgb>
        #include <unpack_4x16>

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDir;
          ndcToCameraRay(
            ndc, inverse(modelMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDir);
          rayDir = normalize(rayDir);

          // .x = Y' of the Y'UV color
          // .y = UV of the Y'UV color, packed as 2 x float16
          // .z = current Z depth for raycasting
          // .w = accumulated density in the 0..INF range
          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          if (frameId == 0) {
            rayData.w = 0.;
            #if !SHOW_COST
              rayData.x = 0.;
              rayData.y = PACK_2x16(vec2(0));
            #endif
          }

          // z < 0      the current point is empty, NextSplatMaterial will find the next splat
          // z > INF    all splats have been blended
          // w > 7.0    the accumulated density leaves no room for further splats since exp(-w) < 0.001
          if (rayData.z < 0. || rayData.z >= INFINITY || rayData.w > 7.0) {
            gl_FragColor = rayData;
            return;
          }

          vec3 yuv = vec3(rayData.x, UNPACK_2x16(rayData.y)); // Y'UV
          vec4 color = vec4(YUV_RGB*yuv, rayData.a); // RGBA

          bvhRay.dir = lightDir; // (vec4(lightDir, 0) * inverse(cameraWorldMatrix)).xyz;
          bvhRay.origin = rayOrigin + rayDir * abs(rayData.z);
          bvhSearchSplats( bvh );

          if (bvhSumColor.w > 0.) {
            float luminance = gsd.brightness;

            #if USE_SHADOWS
              luminance *= exp(-bvhSumShadow);
              luminance += 0.2; // ambient occlusion (AO) or global illumination (GI)
            #endif

            vec4 vol = bvhSumColor * RAY_STEP;
            vol.rgb *= luminance;
            
            color.rgb += exp(-color.w) * vol.rgb;
            color.w += vol.w;
            
            rayData.z += RAY_STEP;
          } else {
            rayData.z *= -1.; // let NextSplatMaterial find the next splat
          }

          #if SHOW_COST
            vec4 cost = vec4(bvhStats.numLookupsBVH, bvhStats.numLookupsSplats, bvhStats.numSplats, 0);
            cost += UNPACK_4x16(rayData.xy);
            rayData.xy = PACK_4x16(cost);
          #endif

          yuv = RGB_YUV*color.rgb;
          gl_FragColor.x = yuv.x; // Y'
          gl_FragColor.y = PACK_2x16(yuv.yz); // UV
          gl_FragColor.z = rayData.z;
          gl_FragColor.w = color.w;

          #if SHOW_COST
            gl_FragColor.xy = rayData.xy;
          #endif
        }`
    });
  }
}

// Finds the nearest 8 splats, blends them, then repeats the same at the next frame.
// In practice, it's better to use a proper rasterizer: https://sparkjs.dev.
// It's only needed to verify the output of raymarching.
class RaytracingMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.MAX_SPLATS_PER_RAY = params.maxSplatsPerRay;
    this.defines.SHOW_COST = +params.showCost;
    this.needsUpdate = true;
  }

  constructor() {

    super({

      defines: {

        SHOW_COST: 0,
        MAX_SPLATS_PER_RAY: 1,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },
        pixelData: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelMatrix: { value: new THREE.Matrix4() },

        lightDir: { value: lightDir },
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

        #include <ray_utils>
        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;

        #include <bvh_sorted_splats>
        
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}

        uniform sampler2D pixelData;

        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        
        uniform vec3 lightDir;
        uniform int frameId;

        #include <common>
        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <gaussian_utils>
        
        // The typical rasterizer-like blending where each splat is approximated with a flat ellipse.
        float raycastSplats(vec3 rayOrigin, vec3 rayDir, inout vec4 rgba) {
          bvhRay = BVHRay( rayOrigin, rayDir );
          bvhSearchSplats( bvh );

          float eps = 0.;

          #if SHOW_COST
            
            vec4 cost = vec4(bvhStats.numLookupsBVH, bvhStats.numLookupsSplats, gNumSplats, 0);
            // rgba.zw are reserved for raytracing metadata
            cost += UNPACK_4x16(rgba.xy);
            rgba.xy = PACK_4x16(cost);

          #else
            
            for (int i = 0; i < gNumSplats; i++) {
              uint splatId = gSplatIds[i];
              vec4 color = texelFetch1D(gsd.splatColors, splatId);
              vec4 splat = texelFetch1D(bvh.position, splatId);

              float scale = splat.w/gsd.splatScale*sqrt(2.0); // = original radius * sqrt(2)
              float dist = gSplatDists[i];
              vec3 midpoint = rayOrigin - splat.xyz + rayDir*dist;

              // beware of float32 accuracy
              eps = max(dist, splat.w)/1e6;

              // .rgb = emission
              // .w = absorption
              color.w *= gsd.splatOpacity;

              #if USE_GAMMA
                color.rgb *= color.rgb; // blend RGB^2, then output sqrt(RGB)
              #endif

              color.rgb *= gsd.brightness;

              // total gaussian density within the segment of the ray
              color.w *= gaussian3d(midpoint/scale);

              color.rgb *= 1. - exp(-color.w); // color.w
              rgba.rgb += color.rgb * exp(-rgba.w);
              rgba.w += color.w;

              if (rgba.w > 7.0)
                return INFINITY;
            }

          #endif
          
          if (gNumSplats < MAX_SPLATS_PER_RAY)
            return INFINITY;
          
          return eps + gSplatDists[gNumSplats - 1];
        }

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDirection;
          ndcToCameraRay(
            ndc, inverse(modelMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDirection);
          rayDirection = normalize(rayDirection);

          vec2 size = vec2(textureSize(pixelData, 0));

          // .x = Y' of the Y'UV color
          // .y = UV of the Y'UV color, packed as 2 x float16
          // .z = current Z depth for raycasting
          // .a = opacity of the accumulated color in the 0..1 range
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          vec3 yuv = vec3(rayData.x, UNPACK_2x16(rayData.y)); // Y'UV
          vec4 color = vec4(YUV_RGB*yuv, rayData.a); // RGBA

          #if SHOW_COST
            color.xy = rayData.xy;
          #endif

          if (frameId == 0) {
            color = vec4(0);
            rayData.z = 0.;
          }
          
          if (rayData.z < INFINITY) {
            rayOrigin += rayData.z * rayDirection;
            float d = raycastSplats(rayOrigin, rayDirection, color);
            rayData.z += d;
            rayOrigin += d*rayDirection;
          }

          yuv = RGB_YUV*color.rgb;
          gl_FragColor.x = yuv.x; // Y'
          gl_FragColor.y = PACK_2x16(yuv.yz); // UV
          gl_FragColor.z = rayData.z;
          gl_FragColor.a = color.a;

          #if SHOW_COST
            gl_FragColor.xy = color.xy;
          #endif
        }`
    });
  }
}

// Draws to the screen.
class OutputMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        showCost: { value: true },
        showProgress: { value: false },
        pixelData: { value: null },
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
        uniform bool showCost;
        uniform bool showProgress;

        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <gsplats_data>
        ${BVHShaderGLSL.common_functions}

        void main() {
          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 o = texelFetch(pixelData, ivec2(vUv*size), 0);

          if (showProgress) {
            o.rgb = o.z >= INFINITY ? vec3(0) : // done rendering
              o.z < 0. ? vec3(1,3,9)*exp2(-abs(o.z)) : // searching for next splat
              vec3(9,3,1)*exp(-o.w); // the remaining density
          } else if (showCost) {
            vec4 cost = UNPACK_4x16(o.xy);
            cost /= 1e5;
            o = mat4x4(9,3,1,0, 3,1,9,0, 1,9,3,0, 3,9,1,0) * cost;
          } else {
            vec3 yuv = vec3(o.x, UNPACK_2x16(o.y)); // Y'UV
            o.rgb = YUV_RGB * yuv;
            #if USE_GAMMA
              o.rgb = sqrt(o.rgb); // gamma correction
            #endif
            //o.rgb += exp(-o.w) * background;
          }

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
  renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0, 0);
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();

  let light = new THREE.DirectionalLight(0xFFFFFF);
  light.position.set(lightDir);
  scene.add(light);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 50);
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

  outputPass = new FullScreenQuad(new OutputMaterial());
  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();
  nextSplatPass = new FullScreenQuad(new NextSplatMaterial());
  nextSplatPass.material.updateDefines();
  raymarchingPass = new FullScreenQuad(new RaymarchingMaterial());
  raymarchingPass.material.updateDefines();

  initGeometry();
  rebuildGUI();

  updateRenderSize();
  window.addEventListener('resize',
    () => updateRenderSize(), false);
}

function updateRenderSize() {
  let w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  renderer.setSize(w, h);
  clearRenderTargets();
}

function clearRenderTargets() {
  let w = window.innerWidth;
  let h = window.innerHeight;
  pixelsRT1.setSize(w, h);
  pixelsRT2.setSize(w, h);
  frameId = 0;
}

async function selectScene() {
  let input = document.createElement('input');
  input.type = 'file';
  input.accept = '*.ply';
  input.multiple = false;
  input.click();
  let blob = await new Promise(resolve =>
    input.onchange = () => resolve(input.files[0]));
  if (!blob) return;

  splatColorsRT?.dispose();
  splatColorsRT = null;
  frameId = -1;

  console.log('Opening file:', (blob.size / 1e6).toFixed(1), 'MB', blob.name);
  let url = URL.createObjectURL(blob);
  await initGeometry(url, blob.name);
  URL.revokeObjectURL(url);

  frameId = 0;
  clearRenderTargets();
  render();
}

async function loadGeometry(url, filename = url) {
  console.time('loadGeometry');
  console.log('Loading scene:', filename);

  let geometry = filename.endsWith('.ply') ?
    await loadPLY(url) :
    await loadGLTF(url);

  geometry.center();
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
  pointCloud.matrixAutoUpdate = false;
  scene.add(pointCloud);

  updateBVHMesh();
  updateSplatColors(pointCloud.geometry);
}

function updateBVHMesh() {
  const bvhGeometry = new THREE.BufferGeometry();
  const attributes = pointCloud.geometry.attributes;

  const index = [];
  const position = attributes.position.clone();
  const count = position.count;

  for (let i = 0; i < count; i++)
    if (i % (1 << params.sparsity) == 0)
      index.push(i, i, i);

  outputContainer.textContent = (index.length / 3) + ' splats';

  const baseScale = 2 ** params.splatScale;
  const scaleAttr = attributes.scale ? attributes.scale.clone() :
    new THREE.BufferAttribute(new Float32Array(count), 1);

  if (attributes.scale && Number.isFinite(scaleAttr.array[0])) {
    for (let i = 0; i < scaleAttr.array.length; i++)
      scaleAttr.array[i] = baseScale * Math.exp(scaleAttr.array[i]);
  } else {
    scaleAttr.array.fill(baseScale * 0.0025);
  }

  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position);
  bvhGeometry.setAttribute('scale', scaleAttr); // this is for computeTriangleBounds
  bvhGeometry.computeBoundsTree(getBVHOptions());

  scene.remove(bvhHelper);
  bvhMesh = new THREE.Mesh(bvhGeometry, new THREE.MeshBasicMaterial());
  bvhHelper = new MeshBVHHelper(bvhMesh, params.depth);
  bvhHelper.displayParents = true;
  bvhHelper.opacity = 0.1;
  bvhHelper.update();
  scene.add(bvhHelper);

  updateBVH();
}

function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();
  gui.onChange((e) => {
    if (e.property != 'showProgress' && e.property != 'render')
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
    updateBVHMesh();
  });
  pointsFolder.add(params, 'sparsity', 0, 16, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'raytracing', 'raymarching']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raytracing') {
    displayFolder.add(params, 'maxSplatsPerRay', 1, 32, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
  }

  if (params.mode === 'raymarching') {
    displayFolder.add(params, 'rayStep', -5, -1, 0.5).onChange(() => {
      nextSplatPass.material.updateDefines();
      raymarchingPass.material.updateDefines();
    });
    displayFolder.add(params, 'shadows').onChange(() => {
      if (params.shadows) frameId = 0;
      raymarchingPass.material.updateDefines();
    });
  }

  if (params.mode == 'raytracing' || params.mode == 'raymarching') {
    pointsFolder.add(params, 'splatOpacity', -3, 10, 0.5);
    pointsFolder.add(params, 'brightness', -5, 5, 0.25);
    pointsFolder.add(params, 'splatScale', 0, 3, 0.25).onChange(() => {
      updateBVHMesh();
    });
    displayFolder.add(params, 'showProgress');
    displayFolder.add(params, 'showCost').onChange(() => {
      raytracingPass.material.updateDefines();
      nextSplatPass.material.updateDefines();
      raymarchingPass.material.updateDefines();
    });
  }
}

function updateBVH() {
  console.time('updateBVH');
  let geometry = bvhMesh.geometry;
  bvh = new MeshBVH(geometry, getBVHOptions());

  // (position.xyz, scale.x) -> position.xyzw
  let { position, scale } = geometry.attributes;

  let count = position.count;
  let position4 = new THREE.BufferAttribute(new Float32Array(count * 4), 4);

  for (let i = 0; i < count; i++) {
    position4.array[i * 4 + 3] = scale.array[i * scale.itemSize];
    for (let j = 0; j < 3; j++)
      position4.array[i * 4 + j] = position.array[i * position.itemSize + j];
  }

  // It would be better if MeshBVH supported
  // the 'position' attribute with 4 elements (xyzw).
  position.copy(position4);

  let bbox = new THREE.Box3();
  bvh.getBoundingBox(bbox);
  let dx = bbox.max.x - bbox.min.x;
  let dy = bbox.max.y - bbox.min.y;
  let dz = bbox.max.z - bbox.min.z;
  console.log('Bounding box:', dx.toFixed(1), 'x', dy.toFixed(1), 'x', dz.toFixed(1));

  console.timeEnd('updateBVH');
}

function updateSplatColors(geometry) {
  let attributes = geometry.attributes;

  let rgb = new FloatVertexAttributeTexture();
  let f_dc = new FloatVertexAttributeTexture();
  let opacity = new FloatVertexAttributeTexture();

  let dummyAttr = new THREE.BufferAttribute(
    new Float32Array(attributes.position.count), 1);
  dummyAttr.array.fill(Number.NaN);

  rgb.updateFrom(attributes.rgb || dummyAttr); // 0..255, uint8
  f_dc.updateFrom(attributes.f_dc || dummyAttr);
  opacity.updateFrom(attributes.opacity || dummyAttr);

  if (!splatColorsRT) {
    let { width, height } = opacity.image;
    splatColorsRT = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType });
  }

  let shader = new FullScreenQuad(new SplatColorsMaterial());
  shader.material.uniforms.ply.value = { f_dc, rgb, opacity };
  renderer.setRenderTarget(splatColorsRT);
  shader.render(renderer);

  f_dc.dispose();
  opacity.dispose();
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

    if (params.mode == 'raytracing') {
      uniforms = raytracingPass.material.uniforms;
      uniforms.bvh.value.updateFrom(bvh);
      uniforms.frameId.value = frameId;
      uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
      uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
      uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
      uniforms.pixelData.value = pixelsRT1.texture;
      uniforms.gsd.value = new GSplatsDataUniformStruct();
      renderer.setRenderTarget(pixelsRT2);
      raytracingPass.render(renderer);
    }

    if (params.mode == 'raymarching') {
      uniforms = nextSplatPass.material.uniforms;
      uniforms.bvh.value.updateFrom(bvh);
      uniforms.gsd.value = new GSplatsDataUniformStruct();
      uniforms.pixelData.value = pixelsRT1.texture;
      uniforms.frameId.value = frameId;
      uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
      uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
      uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
      renderer.setRenderTarget(pixelsRT2);
      nextSplatPass.render(renderer);

      [pixelsRT1, pixelsRT2] = [pixelsRT2, pixelsRT1];

      uniforms = raymarchingPass.material.uniforms;
      uniforms.bvh.value.updateFrom(bvh);
      uniforms.gsd.value = new GSplatsDataUniformStruct();
      uniforms.pixelData.value = pixelsRT1.texture;
      uniforms.frameId.value = frameId;
      uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
      uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
      uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
      renderer.setRenderTarget(pixelsRT2);
      raymarchingPass.render(renderer);
    }

    uniforms = outputPass.material.uniforms;
    uniforms.showCost.value = params.showCost;
    uniforms.showProgress.value = params.showProgress;
    uniforms.pixelData.value = pixelsRT2.texture;
    renderer.setRenderTarget(null);
    outputPass.render(renderer);

    [pixelsRT1, pixelsRT2] = [pixelsRT2, pixelsRT1];
    frameId++;
  }
}

init();
render();
