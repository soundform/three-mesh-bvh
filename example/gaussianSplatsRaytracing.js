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
  strategy: SAH,
  maxDepth: 32,
  maxLeafTris: 8,
  sparsity: 0,
  splatScale: 1.5, // exp2, same as maxStdDev in https://sparkjs.dev
  splatOpacity: 0, // exp2, density that absorbs light 
  splatBrightness: 0, // exp2, luminance that emits light
  maxSamplesPerSplat: 1,
  maxSplatsPerRay: 8,
  flipY: false,
  shadows: false,
  showCost: false,
};

const getBVHOptions = () => ({
  strategy: params.strategy,
  maxDepth: params.maxDepth,
  maxLeafTris: params.maxLeafTris,
});

let renderer, camera, scene, gui, stats, outputContainer;
let bvh, bvhMesh, bvhHelper, pointCloud;
let raytracingPass, drawPixelsPass, shadowsPass;
let pixelsRT1, pixelsRT2, shadowsDataRT, splatColorsRT;
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
  splatBrightness = 2 ** params.splatBrightness;
  splatColors = splatColorsRT.texture;
  shadowsData = shadowsDataRT?.texture || dummyRT.texture;
}

THREE.ShaderChunk['yuv_rgb'] = /* glsl */`
  const mat3 YUV_RGB = mat3(1,1,1,  0,-0.34,1.77, 1.4,-0.72,0);
  const mat3 RGB_YUV = inverse(YUV_RGB); // yuv = RGB_YUV * rgb
`;

THREE.ShaderChunk['gsplats_data'] = /* glsl */`
  struct GSplatsData {
    int splatsCount;
    
    float splatScale;
    float splatOpacity;
    float splatBrightness;
    
    sampler2D splatColors;
    sampler2D shadowsData;
  };
`;

THREE.ShaderChunk['unpack_4x16'] = /* glsl */`
  #define PACK_2x16(xy)   uintBitsToFloat(packHalf2x16(xy))
  #define PACK_4x16(v)    vec2(PACK_2x16(v.xy), PACK_2x16(v.zw))
  #define UNPACK_2x16(x)  unpackHalf2x16(floatBitsToUint(x))
  #define UNPACK_4x16(v)  vec4(UNPACK_2x16(v.x), UNPACK_2x16(v.y))
`;

THREE.ShaderChunk['raycast_splats'] = /* glsl */`
  #ifndef SHOW_COST
  #define SHOW_COST 0
  #endif

  #ifndef USE_SHADOWS
  #define USE_SHADOWS 0
  #endif
  
  // integrate(exp(-x*x))*2/sqrt(PI) = -1..1
  float erfc(float x) {
      return sign(x)*sqrt(1. - exp2(-1.787776*x*x));
  }

  // integrate(exp(-sqr(pos + dir*t)), t=0..INF)*2/sqrt(PI) = 0..2
  float erfc_3d(vec3 pos, vec3 dir) {
    float b = dot(pos, -dir);       // -INF..INF
    float h = dot(pos, pos) - b*b;  // 0..INF
    float s = 1. + erfc(b);         // 0..2
    return s*exp(-h);
  }

  float raycastSplats(vec3 rayOrigin, vec3 rayDir, inout vec4 rgba) {
    BVHIntersectResult res;
    bvhIntersectSplats( bvh, rayOrigin, rayDir, res );

    float eps = 0.;

    #if SHOW_COST
      
      vec4 cost = vec4(res.numLookupsBVH, res.numLookupsSplats, res.numSplats, 0);
      cost /= float(gsd.splatsCount)/1e2;
      // rgba.zw are reserved for raytracing metadata
      cost += UNPACK_4x16(rgba.xy);
      rgba.xy = PACK_4x16(cost);

    #else
      
      if (res.numSplats == 0)
        return INFINITY;

      vec2[MAX_SPLATS_PER_RAY*2] pts; // sorted by .x

      for (int k = 0; k < res.numSplats; k++) {
        vec2 tt = gSplatDists[k];

        for (int i = 0; i < 2; i++) {
          vec2 pt = vec2(tt[i], k+1);
          pt.y *= i == 0 ? +1. : -1.;
          pts[k*2+i] = pt;

          for (int j = k*2+i; j > 0; j--) {
            if (pts[j].x >= pts[j-1].x)
              break;
            pts[j] = pts[j-1];
            pts[j-1] = pt;
          }
        }
      }

      int bitmask = 0;

      for (int k = 0; k < res.numSplats*2 - 1; k++) {
        int b = int(pts[k].y);
        if (b > 0) bitmask |=  (1 << (+b - 1));
        if (b < 0) bitmask &= ~(1 << (-b - 1));
        if (bitmask == 0) continue;

        vec2 segment = vec2(pts[k].x, pts[k+1].x);

        for (int i = 0; i < res.numSplats; i++) {
          if ((bitmask & (1 << i)) == 0)
            continue;

          uint splatId = gSplatIds[i];
          vec4 color = texelFetch1D(gsd.splatColors, splatId);
          vec4 splat = texelFetch1D(bvh.position, splatId);
          float radius = splat.w;

          vec2 seg = raySphere(rayOrigin - splat.xyz, rayDir, radius);
          seg.x = max(seg.x, segment.x);
          seg.y = min(seg.y, segment.y);
          if (seg.x >= seg.y) continue;

          float scale = radius/gsd.splatScale*sqrt(2.0);
          float dist = dot(vec2(0.5), segment); // gSplatDists[i];
          vec3 midpoint = rayOrigin - splat.xyz + rayDir*dist;

          // beware of float32 accuracy
          eps = max(dist, radius)/1e6;

          // .rgb = emission
          // .a = absorption
          color.rgb *= color.rgb; // blend RGB^2, then output sqrt(RGB)
          color.w *= gsd.splatOpacity;

          #if USE_SHADOWS

            float luminance = texelFetch1D(gsd.shadowsData, splatId).x;
            float fog = erfc_3d(midpoint/scale, lightDir)*0.5; // 0..1
            color.rgb *= luminance * exp(-fog * color.w);

          #endif

          // total gaussian density within the segment of the ray
          //color.w *= exp(-dot(midpoint/scale, midpoint/scale));
          float fx = erfc_3d((rayOrigin - splat.xyz + rayDir*seg.x)/scale, rayDir);
          float fy = erfc_3d((rayOrigin - splat.xyz + rayDir*seg.y)/scale, rayDir);
          color.w *= max(fx - fy, 0.)*0.5; // 0..1

          color.rgb *= color.w;
          rgba.rgb += color.rgb * exp(-rgba.w);
          rgba.w += color.w;

          if (rgba.w > 4.0)
            return INFINITY;
        }
      }

    #endif
    
    if (res.numSplats < MAX_SPLATS_PER_RAY)
      return INFINITY;
    return eps + dot(vec2(0.5), gSplatDists[res.numSplats - 1]);
  }
`;

class RaytracingMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.BVH_STACK_DEPTH = params.maxDepth;
    this.defines.MAX_SPLATS_PER_RAY = params.maxSplatsPerRay;
    this.defines.SHOW_COST = +params.showCost;
    this.defines.USE_SHADOWS = +params.shadows;
    this.defines.MAX_SAMPLES_PER_SPLAT = params.maxSamplesPerSplat;
    this.needsUpdate = true;
  }

  constructor(params) {

    super({

      defines: {

        SHOW_COST: 0,
        USE_SHADOWS: 1,
        MAX_RAYCASTS: 1,
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
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}
        
        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;
        uniform sampler2D pixelData;

        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        
        uniform vec3 lightDir;
        uniform int frameId;

        #include <common>
        #include <yuv_rgb>
        #include <unpack_4x16>
        #include <raycast_splats>

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

            for (int step = 0; step < MAX_RAYCASTS; step++) {
              float d = raycastSplats(rayOrigin, rayDirection, color);
              rayData.z += d;
              rayOrigin += d*rayDirection;
              if (d >= INFINITY) break;
            }
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

    this.setValues(params);
  }
}

class ComputeShadowsMaterial extends THREE.ShaderMaterial {

  constructor() {

    super({

      defines: {

        USE_SHADOWS: 0,
        BVH_STACK_DEPTH: params.maxDepth,
        MAX_SPLATS_PER_RAY: 16,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        gsd: { value: null },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelMatrix: { value: new THREE.Matrix4() },

        lightDir: { value: lightDir },

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
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}

        #include <gsplats_data>

        uniform BVH bvh;
        uniform GSplatsData gsd;

        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        
        uniform vec3 lightDir;

        #include <common>
        #include <unpack_4x16>
        #include <raycast_splats>

        void main() {
          ivec2 size = textureSize(bvh.position, 0);
          ivec2 uv = ivec2(vUv * vec2(size));
          uint splatId = uint(uv.x + uv.y * size.x);
          vec4 splat = texelFetch1D( bvh.position, splatId );

          vec3 dir = lightDir; // (vec4(lightDir, 0) * inverse(cameraWorldMatrix)).xyz;

          gl_FragColor = vec4(0);
          raycastSplats(splat.xyz + dir*splat.w, dir, gl_FragColor);
          gl_FragColor.x = max(exp(-gl_FragColor.w), 0.0);
          gl_FragColor.x += 0.2; // this should be ambient occlusion (AO) or global illumination (GI)
        }`
    });
  }
}

class DrawPixelsMaterial extends THREE.ShaderMaterial {
  constructor(params) {
    super({
      uniforms: {
        splatBrightness: { value: 1 },
        showCost: { value: true },
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

        uniform float splatBrightness;
        uniform bool showCost;
        uniform sampler2D pixelData;

        #include <yuv_rgb>
        #include <unpack_4x16>

        void main() {
          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 o = texelFetch(pixelData, ivec2(vUv*size), 0);
          
          if (showCost) {
            vec4 cost = UNPACK_4x16(o.xy);
            o = mat4x4(9,3,1,0, 3,1,9,0, 1,9,3,0, 3,9,1,0) * cost;
            o.w = 1.0;
          } else {
            vec3 yuv = vec3(o.x, UNPACK_2x16(o.y)); // Y'UV
            o.rgb = splatBrightness*YUV_RGB*yuv;
            o.rgb = sqrt(o.rgb); // gamma correction
            o.rgb *= 1.0 - exp(-o.w);
            o.w = 1.0;
          }

          if (isnan(dot(o,o)))
            o = vec4(0,1,0,1);

          gl_FragColor = o;
        }`
    });

    this.setValues(params);
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

  let orbit = new OrbitControls(camera, renderer.domElement);
  orbit.addEventListener('change', () => clearRenderTargets());

  pixelsRT1 = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });
  pixelsRT2 = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });

  stats = new Stats();
  document.body.appendChild(stats.dom);

  drawPixelsPass = new FullScreenQuad(new DrawPixelsMaterial());
  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();
  shadowsPass = new FullScreenQuad(new ComputeShadowsMaterial());

  initGeometry();
  rebuildGUI();

  updateRenderSize();
  window.addEventListener('resize',
    () => updateRenderSize(), false);

  document.addEventListener('keypress', (e) => {
    if (e.code == 'Space')
      orbit.enabled = !orbit.enabled;
  });
}

function updateRenderSize() {
  let w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  renderer.setSize(w, h);
  clearRenderTargets();
}

function clearRenderTargets() {
  let w = window.innerWidth, h = window.innerHeight;
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

  shadowsDataRT?.dispose();
  shadowsDataRT = null;
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
  const material = new THREE.PointsMaterial({ color: 0xFFFFFF });
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

  if (params.flipY) {
    let m = position.itemSize;
    for (let i = 0; i < count; i++)
      position.array[i * m + 1] *= -1;
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
  gui.onChange(() => {
    clearRenderTargets();
  });

  gui.add(params, 'open');

  const pointsFolder = gui.addFolder('points');
  pointsFolder.add(params, 'strategy', { CENTER, AVERAGE, SAH }).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree(getBVHOptions());
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateBVH();
  });
  pointsFolder.add(params, 'maxDepth', 4, 64, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.add(params, 'maxLeafTris', 1, 16, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.add(params, 'sparsity', 0, 16, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'splats']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'splats') {
    //displayFolder.add(params, 'maxSamplesPerSplat', 1, 4, 1).onChange(() => {
    //  raytracingPass.material.updateDefines();
    //});
    displayFolder.add(params, 'maxSplatsPerRay', 1, 32, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'splatOpacity', -3.5, 3.5, 0.5);
    displayFolder.add(params, 'splatBrightness', -3, 10, 0.5);
    displayFolder.add(params, 'splatScale', 0, 3, 0.25).onChange(() => {
      updateBVHMesh();
    });
    displayFolder.add(params, 'flipY').onChange(() => {
      updateBVHMesh();
    });
    displayFolder.add(params, 'showCost').onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'shadows').onChange(() => {
      if (params.shadows) frameId = 0;
      raytracingPass.material.updateDefines();
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

  position.copy(position4);
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

function updateShadowsData() {
  let uniforms = shadowsPass.material.uniforms;
  uniforms.bvh.value.updateFrom(bvh);

  if (!shadowsDataRT) {
    let { width, height } = uniforms.bvh.value.position.image;
    shadowsDataRT = new THREE.WebGLRenderTarget(width, height, { format: THREE.RedFormat, type: THREE.HalfFloatType });
  }

  uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
  uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
  uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
  uniforms.gsd.value = new GSplatsDataUniformStruct();
  uniforms.gsd.value.shadowsData = dummyRT.texture;
  renderer.setRenderTarget(shadowsDataRT);
  shadowsPass.render(renderer);
}

function render() {

  stats.update();
  requestAnimationFrame(render);

  if (frameId < 0)
    return;

  if (params.mode === 'points') {

    if (!pointCloud) return;
    pointCloud.material.size = 0.005;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

  } else if (params.mode === 'splats') {
    if (!bvh) return;

    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    if (frameId == 0 && params.shadows && !shadowsDataRT)
      updateShadowsData();

    let uniforms = raytracingPass.material.uniforms;
    uniforms.bvh.value.updateFrom(bvh);
    uniforms.frameId.value = frameId;
    uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
    uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
    uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
    uniforms.pixelData.value = pixelsRT1.texture;
    uniforms.gsd.value = new GSplatsDataUniformStruct();
    renderer.setRenderTarget(pixelsRT2);
    raytracingPass.render(renderer);

    uniforms = drawPixelsPass.material.uniforms;
    uniforms.splatBrightness.value = 2 ** params.splatBrightness;
    uniforms.showCost.value = params.showCost;
    uniforms.pixelData.value = pixelsRT2.texture;
    renderer.setRenderTarget(null);
    drawPixelsPass.render(renderer);

    [pixelsRT1, pixelsRT2] = [pixelsRT2, pixelsRT1];
    frameId++;
  }
}

init();
render();
