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
  regenerate: () => updateBVH(),

  mode: 'points',
  strategy: SAH,
  maxLeafTris: 8,
  sparsity: 0,
  splatSize: -7, // exp2, splat radius
  splatOpacity: 0, // exp2, density that absorbs light 
  splatBrightness: 0, // exp2, luminance that emits light
  maxRaycasts: 1,
  maxSplatsPerRay: 8,
  showCost: false,
};

const getBVHOptions = () => ({ strategy: params.strategy, maxLeafTris: params.maxLeafTris });

let renderer, camera, scene, gui, stats, outputContainer;
let bvh, bvhMesh, bvhHelper, pointCloud;
let raytracingPass, drawPixelsPass, shadowsPass;
let renderTargets = [], shadowsDataRT, splatColorsRT;
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
  splatSize = 2 ** params.splatSize;
  splatOpacity = 2 ** params.splatOpacity;
  splatBrightness = 2 ** params.splatBrightness;
  splatColors = splatColorsRT.texture;
  shadowsData = shadowsDataRT.texture;
}

THREE.ShaderChunk['yuv_rgb'] = /* glsl */`
  const mat3 YUV_RGB = mat3(1,1,1,  0,-0.34,1.77, 1.4,-0.72,0);
  const mat3 RGB_YUV = inverse(YUV_RGB);
`;

THREE.ShaderChunk['gsplats_data'] = /* glsl */`
  struct GSplatsData {
    int splatsCount;
    
    float splatSize;
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
  
  // 2/sqrt(PI)*integrate(exp(-x*x))
  float erfc(float x) {
      return sign(x)*sqrt(1. - exp2(-1.787776*x*x));
  }

  float integrateGaussian(vec3 ro, vec3 rd, float SR) {
    float b = dot(ro, rd);
    float h = dot(ro, ro) - b*b;
    float s = (1. + erfc(b*SR))/SR;
    s *= sqrt(PI)/2.;
    s *= exp(-SR*SR*h);
    return s;
  }

  float raycastSplats(vec3 ro, vec3 rd, inout vec4 rgba) {
    BVHIntersectResult res;
    bvhIntersectSplats( bvh, ro, rd, splatsData.splatSize, res );

    #if SHOW_COST
      
      vec4 cost = vec4(res.numLookupsBVH, res.numLookupsSplats, res.numSplats, 0)/1e3;
      // rgba.zw are reserved for raytracing metadata
      cost += UNPACK_4x16(rgba.xy);
      rgba.xy = PACK_4x16(cost);

    #else

      // blend all splats along the ray
      for (int i = 0; i < res.numSplats; i++) {
        uint splatId = gSplatIds[i];
        vec3 splatPos = texelFetch1D( bvh.position, splatId ).xyz;
        float splatScale = splatsData.splatSize/3.0;
        vec3 pos = ro + rd*gSplatDists[i] - splatPos;
        float density = exp(-dot(pos/splatScale, pos/splatScale));
        //if (length(pos) > splatSize) continue;

        float opacity = splatsData.splatOpacity;
        vec4 splatColor = texelFetch1D(splatsData.splatColors, splatId);
        opacity *= splatColor.a;

        // color.a = the amount of particles along the ray
        vec4 color = vec4(splatColor.rgb, density * opacity);

        #if USE_SHADOWS
          
          color.rgb *= texelFetch1D(splatsData.shadowsData, splatId).x;
          float sum = integrateGaussian(pos, lightDir, 1.0/splatScale);
          color.rgb *= exp(-sum * opacity);
        
        #endif

        // normal blending
        color.rgb *= color.a;
        rgba += color * (1.0 - rgba.a);
        rgba.a = clamp(rgba.a, 0.0, 1.0);
        if (rgba.a > 0.995)
          return INFINITY;
      }

    #endif
    
    if (res.numSplats < MAX_SPLATS_PER_RAY)
      return INFINITY;
    return gSplatDists[res.numSplats - 1];
  }
`;

class RaytracingMaterial extends THREE.ShaderMaterial {

  updateDefines() {
    this.defines.MAX_SPLATS_PER_RAY = params.maxSplatsPerRay;
    this.defines.SHOW_COST = +params.showCost;
    this.defines.MAX_RAYCASTS = params.maxRaycasts;
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
        splatsData: { value: null },
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
        uniform GSplatsData splatsData;
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
          // .y = UV of the Y'UV color, packed as float16
          // .z = Z depth for raycasting
          // .a = opacity of the accumulated color in the 0..1 range
          vec4 rayData = texelFetch(pixelData, ivec2(vUv*size), 0);

          vec3 yuv = vec3(rayData.x, UNPACK_2x16(rayData.y)); // Y'UV
          vec4 color = vec4(YUV_RGB*yuv, rayData.a); // RGBA

          if (frameId == 0) {
            color = vec4(0);
            rayData.z = 0.;
          }
          
          if (rayData.z < INFINITY) {
            rayOrigin += rayData.z * rayDirection;

            for (int step = 0; step < MAX_RAYCASTS; step++) {
              float d = raycastSplats(rayOrigin, rayDirection, color);
              // beware of float32 accuracy
              d += max(d/1e6, splatsData.splatSize/1e4);
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
        }`
    });

    this.setValues(params);
  }
}

class ComputeShadowsMaterial extends THREE.ShaderMaterial {

  constructor(params) {

    super({

      defines: {

        USE_SHADOWS: 0,
        MAX_SPLATS_PER_RAY: 32,

      },

      uniforms: {

        bvh: { value: new MeshBVHUniformStruct() },
        splatsData: { value: null },

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
        uniform GSplatsData splatsData;

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
          vec3 splatPos = texelFetch1D( bvh.position, splatId ).xyz;

          //splatPos = (vec4(splatPos, 1) * modelMatrix).xyz; // ??
          vec3 lightDirCam = (vec4(lightDir, 0) * inverse(cameraWorldMatrix)).xyz;
          splatPos += lightDirCam * splatsData.splatSize;

          gl_FragColor = vec4(0);
          raycastSplats(splatPos, lightDirCam, gl_FragColor);
          gl_FragColor.x = max(1.0 - gl_FragColor.w, 0.0);

          //gl_FragColor.x = float(splatId) / float(splatsData.splatsCount);
        }`
    });

    this.setValues(params);
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
            o.rgb = YUV_RGB*yuv;
            o.rgb *= splatBrightness;
          }

          gl_FragColor = o;
        }`
    });

    this.setValues(params);
  }
}

// (f_dc, opacity) -> rgba
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
        struct PLY { sampler2D f_dc, opacity; };
        uniform PLY ply;

        #include <common>

        void main() {
          vec2 size = vec2(textureSize(ply.f_dc, 0));

          vec3 f_dc = texelFetch(ply.f_dc, ivec2(vUv*size), 0).xyz;
          float opacity = texelFetch(ply.opacity, ivec2(vUv*size), 0).x;

          gl_FragColor.rgb = f_dc/sqrt(PI)*0.5 + 0.5;
          gl_FragColor.a = 1./(1. + exp(-opacity));

          gl_FragColor = max(gl_FragColor, 0.); // should be a no-op
        }`
    });
  }
}

async function init() {
  outputContainer = document.getElementById('output');

  // renderer setup
  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0, 0);
  document.body.appendChild(renderer.domElement);

  scene = new THREE.Scene();

  let light = new THREE.DirectionalLight(0xFFFFFF);
  light.position.set(lightDir);
  scene.add(light);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.001, 50);
  camera.position.set(1, 1, 2);
  camera.far = 100;
  camera.updateProjectionMatrix();

  let orbit = new OrbitControls(camera, renderer.domElement);
  orbit.addEventListener('change', () => resetRenderState());

  for (let i = 0; i < 2; i++)
    renderTargets[i] = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });

  stats = new Stats();
  document.body.appendChild(stats.dom);

  drawPixelsPass = new FullScreenQuad(new DrawPixelsMaterial());
  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();
  shadowsPass = new FullScreenQuad(new ComputeShadowsMaterial());

  initGeometry();
  rebuildGUI();
  updateRenderSize();

  window.addEventListener('resize', updateRenderSize, false);
}

function updateRenderSize() {
  let w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  renderer.setSize(w, h);
  resetRenderState();
}

function resetRenderState() {
  let w = window.innerWidth, h = window.innerHeight;

  renderTargets[0].setSize(w, h);
  renderTargets[1].setSize(w, h);

  frameId = 0;
}

async function loadGeometry() {
  console.time('loadGeometry');
  console.log('Loading scene:', sceneFile);
  let geometry;

  if (sceneFile.endsWith('.ply')) {
    let ply = new PLYLoader();

    // geometry.attributes
    ply.setCustomPropertyNameMapping({
      scale: ['scale_0', 'scale_1', 'scale_2'], // scale = log(S)
      f_dc: ['f_dc_0', 'f_dc_1', 'f_dc_2'], // f_dc = (RGB - 0.5)*sqrt(PI)*2.0
      opacity: ['opacity'], // opacity = -log(1.0/A - 1.0), A=0..1
      // rot: ['rot_0', 'rot_1', 'rot_2', 'rot_3'], // quaternion rotation
    });

    geometry = await ply.loadAsync(sceneFile);
  } else {
    let gltf = await new GLTFLoader()
      .setMeshoptDecoder(MeshoptDecoder)
      .loadAsync(sceneFile);
    gltf.scene.updateMatrixWorld(true);
    let gltfMesh = gltf.scene.children[0];
    geometry = gltfMesh.geometry;
  }

  geometry.center();
  console.timeEnd('loadGeometry');
  return geometry;
}

async function initGeometry() {
  const geometry = await loadGeometry();
  const material = new THREE.PointsMaterial({ color: 0xFFFFFF });
  pointCloud = new THREE.Points(geometry, material);
  pointCloud.name = 'Point Cloud';
  pointCloud.matrixAutoUpdate = false;
  scene.add(pointCloud);

  updateBVHMesh();
  updateSplatColors();
}

function updateBVHMesh() {
  const bvhGeometry = new THREE.BufferGeometry();
  const { position } = pointCloud.geometry.attributes;
  const index = [];
  for (let i = 0; i < position.count; i++)
    if (i % (1 << params.sparsity) == 0)
      index.push(i, i, i);
  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position);
  bvhGeometry.computeBoundsTree(getBVHOptions());
  outputContainer.textContent = (index.length / 3) + ' splats';

  scene.remove(bvhHelper);
  bvhMesh = new THREE.Mesh(bvhGeometry, new THREE.MeshBasicMaterial({ color: 0xff0000 }));
  bvhHelper = new MeshBVHHelper(bvhMesh, params.depth);
  bvhHelper.name = 'BVH Helper';
  scene.add(bvhHelper);

  updateBVH();
}

function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();
  gui.onChange(() => {
    resetRenderState();
  });

  const pointsFolder = gui.addFolder('points');
  pointsFolder.add(params, 'strategy', { CENTER, AVERAGE, SAH }).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree(getBVHOptions());
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateBVH();
  });
  pointsFolder.add(params, 'maxLeafTris', 1, 16, 1).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree(getBVHOptions());
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateBVH();
  });
  pointsFolder.add(params, 'sparsity', 0, 12, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'gaussians']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'gaussians') {
    //displayFolder.add(params, 'maxRaycasts', 1, 16, 1).onChange(() => {
    //  raytracingPass.material.updateDefines();
    //});
    displayFolder.add(params, 'maxSplatsPerRay', 1, 64, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'splatOpacity', -5, 5, 0.5);
    displayFolder.add(params, 'splatBrightness', -5, 5, 0.5);
    displayFolder.add(params, 'splatSize', -10, 0, 0.5);
    displayFolder.add(params, 'showCost').onChange(() => {
      raytracingPass.material.updateDefines();
    });
  }
}

function updateBVH() {
  if (!params.sparsity) console.time('MeshBVH');
  bvh = new MeshBVH(bvhMesh.geometry, getBVHOptions());
  if (!params.sparsity) console.timeEnd('MeshBVH');
  rebuildGUI();
}

function updateSplatColors() {
  if (splatColorsRT) return;

  let attributes = pointCloud.geometry.attributes;

  let f_dc = new FloatVertexAttributeTexture();
  let opacity = new FloatVertexAttributeTexture();

  f_dc.updateFrom(attributes.f_dc);
  opacity.updateFrom(attributes.opacity);

  let { width, height } = f_dc.image;
  splatColorsRT = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType });

  let shader = new FullScreenQuad(new SplatColorsMaterial());
  shader.material.uniforms.ply.value = { f_dc, opacity };
  renderer.setRenderTarget(splatColorsRT);
  shader.render(renderer);

  f_dc.dispose();
  opacity.dispose();

  renderer.setRenderTarget(null);
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
  uniforms.splatsData.value = new GSplatsDataUniformStruct();
  uniforms.splatsData.value.shadowsData = dummyRT.texture;
  renderer.setRenderTarget(shadowsDataRT);
  shadowsPass.render(renderer);
}

function render() {

  stats.update();
  requestAnimationFrame(render);

  if (params.mode === 'points') {

    if (!pointCloud) return;
    pointCloud.material.size = 0.5 * 2 ** params.splatSize;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);

  } else if (params.mode === 'gaussians') {
    if (!bvh) return;

    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    if (frameId == 0)
      updateShadowsData();

    let uniforms = raytracingPass.material.uniforms;
    uniforms.bvh.value.updateFrom(bvh);
    uniforms.frameId.value = frameId;
    uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
    uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
    uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
    uniforms.pixelData.value = renderTargets[0].texture;
    uniforms.splatsData.value = new GSplatsDataUniformStruct();
    renderer.setRenderTarget(renderTargets[1]);
    raytracingPass.render(renderer);

    uniforms = drawPixelsPass.material.uniforms;
    uniforms.splatBrightness.value = 2 ** params.splatBrightness;
    uniforms.showCost.value = params.showCost;
    uniforms.pixelData.value = renderTargets[1].texture;
    renderer.setRenderTarget(null);
    drawPixelsPass.render(renderer);

    renderTargets = [renderTargets[1], renderTargets[0]];
    frameId++;
  }
}

init();
render();
