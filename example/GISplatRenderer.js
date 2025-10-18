import * as THREE from 'three';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { BVHShaderGLSL, MeshBVHUniformStruct } from 'three-mesh-bvh';

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
  int gTexLookups = 0;

  void initFogSplat(mat4 modelWorldMatrix) {
    gFogSplat = vec4(0, -5, 0, gsd.maxStdDev);
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

  void bvhInitSearch() {
    gSumShadow = 0.;
    gSumColor = vec4(0);
    gTexLookups = 0;

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
        gTexLookups++;
        gSunDist = delta * (bb.z - aa.z)/gSunDir.z;
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
        bool skipFog(vec3 rayOrigin, vec3 rayDir, inout float zDepth, inout vec4 color) {
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

        bool blendSplats(vec3 rayOrigin, vec3 rayDir, inout PixelData pd) {
          bvhSearchSplats( bvh );
          
          pd.cost += bvhTexLookups + gTexLookups;

          if (gSumColor.w <= 0. && gFogSplat.w == 0.)
            return false;

          float luminance = gsd.brightness;

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

          pd.zDepth += RAY_STEP;
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

          if (gsd.fogDensity > 0. && skipFog(rayOrigin, rayDir, pd.zDepth, pd.color)) {
            // ...
          } else if (!blendSplats(rayOrigin, rayDir, pd)) {
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

export function updateShadowMapGI(renderer, params, shadowMapRT, bvh, gsd, pointCloud) {
  console.time('Update shadowMap');

  let size = 2048, layers = params.shadowMapLayers;
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
