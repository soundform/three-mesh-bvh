import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import Stats from 'stats.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

import {
  computeBoundsTree, disposeBoundsTree,
  SAH,
  BVHShaderGLSL,
  MeshBVHUniformStruct,
} from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

const params = {
  open: () => loadSceneFile(),
  //size: () => [16 * 150, 16 * 150],
  size: () => [window.innerWidth, window.innerHeight],

  mode: 'points',
  render: true,
  strategy: SAH,
  maxDepth: 40,
  maxLeafTris: 8,
  sparsity: 0,
  maxStdDev: 1.5, // exp2, same as maxStdDev in https://sparkjs.dev
  splatScale: 0, // exp2
  splatOpacity: 0, // exp2, density that absorbs light 
  brightness: 0, // exp2, brightness of sunlight or of the splats themselves
  ambientLight: -6, // exp2
  rayStep: -2.0, // exp10
  fogDensity: -3.0, // exp10
  shadows: false,
  monochrome: false,
  lightPos: new THREE.Vector3(8e3, 1e3, 9e3),
  shadowMapLayers: 16,

  get bvhOptions() {
    return {
      strategy: params.strategy,
      maxDepth: params.maxDepth,
      maxLeafTris: params.maxLeafTris,
    };
  },
};

import {
  GSplatsDataUniformStruct,
  DoubleBufferRenderTarget,
  loadPLY,
  getSunMatrix4,
  updateBVH,
  disposeBVH,
  updateSplatColors,
  updateShadowMapGI,
  runRaymarchingPass,
  runRenderPass,
  updateShaderDefines,
} from './GISplatRenderer.js';

let renderer, camera, scene, orbit, gui, stats, outputContainer;
let bvh, pointCloud, raytracingPass;
let pixelsRT = new DoubleBufferRenderTarget();
let frameId = 0;

//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';
const sceneFile = 'models/soundform.ply';

// Finds the nearest 8 splats, blends them, then repeats the same at the next frame.
// In practice, it's usually better to use a proper rasterizer: https://sparkjs.dev.
class RaytracingMaterial extends THREE.ShaderMaterial {

  updateDefines(params) {
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
        float gDistScale = 1.0;
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
          float h = dot(r, r) - t*t;

          // TODO: Multiply sqrt(h) by the distance from the screen.
          // True rendering needs to capture all splats that map
          // to a pixel, not just those that intersect with a ray.
          if (h >= gDistScale || t <= 0. || t*splat.w >= gSplats[3].z)
            return false;          

          vec4 a = gSplats[0]; // a.x <= a.z <= b.x
          vec4 b = gSplats[1]; // b.x <= b.z <= c.x
          vec4 c = gSplats[2]; // c.x <= c.z <= d.x
          vec4 d = gSplats[3]; // d.x <= d.z
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

        ///// splat color blending ///////////////////////////////////////////
        
        bool blendSplat(vec2 entry, inout vec4 sumColor) {
          uint splatId = uint(entry.y);
          float dist = entry.x;

          if (dist >= INFINITY || sumColor.w >= 1.0)
            return false;
          
          vec4 splat = texelFetch1D(bvh.position, splatId);
          vec4 color = texelFetch1D(gsd.splatColors, splatId);

          #if USE_GAMMA
            color.rgb *= color.rgb; // blend RGB^2, then output sqrt(RGB)
          #endif

          color.w *= gsd.splatOpacity;
          color.w /= gDistScale;

          // rasterizer-style blending: splats are approximated with flat ellipses
          vec3 r = (gRayOrigin + bvhRayDir*dist - splat.xyz) / splat.w;
          color.w *= gaussian3d(r * gsd.maxStdDev / SQRT_2 / gDistScale);

          color.rgb *= color.w;
          sumColor += (1. - sumColor.w) * color;
          return true;
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
          //gDistScale = 1.0 + pd.zDepth/length(rayOrigin);
          bvhSearchSplats( bvh );
          pd.cost += bvhTexLookups; // total cost
          vec4 rgba = pd.color;

          blendSplat(gSplats[0].xy, rgba) &&
          blendSplat(gSplats[0].zw, rgba) &&
          blendSplat(gSplats[1].xy, rgba) &&
          blendSplat(gSplats[1].zw, rgba) &&
          blendSplat(gSplats[2].xy, rgba) &&
          blendSplat(gSplats[2].zw, rgba) &&
          blendSplat(gSplats[3].xy, rgba) &&
          blendSplat(gSplats[3].zw, rgba);
          
          pd.color = rgba;
          pd.zDepth += (1. + 1e-6) * gSplats[3].z + 1e-6;
          gl_FragColor = packPixelData(pd);
        }`
    });
  }
}

async function init() {
  outputContainer = document.getElementById('output');

  let [w, h] = params.size();
  console.log('Canvas size:', w, 'x', h);

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

  stats = new Stats();
  document.body.appendChild(stats.dom);

  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines(params);

  window.params = params;
  window.THREE = THREE;
  window.renderer = renderer;

  initGeometry();
  rebuildGUI();

  updateRenderSize();
  window.addEventListener('resize',
    () => updateRenderSize(), false);
}

async function sleep(msec) {
  return new Promise(resolve => setTimeout(resolve, msec));
}

async function updateBVHMesh() {
  console.time('updateBVH');
  outputContainer.textContent = 'Updating BVH...';
  await sleep(0);

  bvh = updateBVH(params, pointCloud, scene);

  let bbox = new THREE.Box3();
  bvh.getBoundingBox(bbox);
  let dx = bbox.max.x - bbox.min.x;
  let dy = bbox.max.y - bbox.min.y;
  let dz = bbox.max.z - bbox.min.z;
  let aabb = dx.toFixed(2) + ' x ' + dy.toFixed(2) + ' x ' + dz.toFixed(2);

  console.timeEnd('updateBVH');
  let n = pointCloud.geometry.attributes.position.count;
  let str = n < 1e3 ? n :
    n > 1e6 ? (n / 1e6).toFixed(1) + 'K' :
      (n / 1e3).toFixed(0) + 'K';
  outputContainer.textContent = str + ' splats | ' + aabb;

  await updateShadowMap();
}

async function updateShadowMap() {
  if (!params.shadows)
    return;

  let gsd = new GSplatsDataUniformStruct(params);
  updateShadowMapGI(renderer, params, bvh, gsd, pointCloud);
  clearRenderTargets();
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
  pixelsRT.setSize(w, h);
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

async function loadGLTF(url) {
  let gltf = await new GLTFLoader()
    .setMeshoptDecoder(MeshoptDecoder)
    .loadAsync(url);
  gltf.scene.updateMatrixWorld(true);
  let gltfMesh = gltf.scene.children[0];
  return gltfMesh.geometry;
}

async function initGeometry(url = sceneFile, filename) {
  disposeBVH();

  const geometry = await loadGeometry(url, filename);
  const material = new THREE.PointsMaterial({ color: 0xCCCCCC });
  scene.remove(pointCloud);
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  let sunMatrix = getSunMatrix4(params.lightPos);
  console.debug('det(sunMatrix) = ' + sunMatrix.determinant().toFixed(2));
  pointCloud.geometry.applyMatrix4(sunMatrix.clone().invert());
  pointCloud.matrix = sunMatrix;
  pointCloud.matrixAutoUpdate = false;
  pointCloud.updateMatrixWorld();

  await updateSplatColors(renderer, pointCloud);
  await updateBVHMesh();
}

function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();
  gui.onChange((e) => {
    if (e.property != 'render' && e.property != 'monochrome' && e.property != 'brightness')
      clearRenderTargets();
  });

  gui.add(params, 'open');

  gui.add(params, 'render').onChange(() => {
    orbit.enabled = params.render;
  });

  const pointsFolder = gui.addFolder('points');

  pointsFolder.add(params, 'maxDepth', 4, 64, 1).onChange(() => {
    updateShaderDefines(params);
    updateBVHMesh();
  });
  pointsFolder.add(params, 'sparsity', 0, 16, 1).onChange(() => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'raytracing', 'raymarching']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raymarching') {
    displayFolder.add(params, 'rayStep', -3, -1, 0.5).onChange(() => {
      updateShaderDefines(params);
    });
    displayFolder.add(params, 'fogDensity', -3.0, 7.5, 0.25).onChange(() => {
      updateShadowMap();
    });
    displayFolder.add(params, 'ambientLight', -10, -1, 0.5);
    displayFolder.add(params, 'shadows').onChange(() => {
      updateShaderDefines(params, 'shadows');
      updateShadowMap();
    });
    displayFolder.add(params, 'shadowMapLayers', 1, 32, 1).onChange(() => {
      updateShadowMap();
    });
  }

  if (params.mode == 'raytracing' || params.mode == 'raymarching') {
    displayFolder.add(params, 'maxStdDev', 0, 3, 0.5).onChange(() => {
      updateBVHMesh();
    });
    displayFolder.add(params, 'splatScale', -4, 4, 0.5).onChange(() => {
      updateBVHMesh();
    });
    displayFolder.add(params, 'splatOpacity', -4, 8, 0.5).onChange(() => {
      updateShadowMap();
    });
    displayFolder.add(params, 'brightness', -3, 3, 0.5);
    displayFolder.add(params, 'monochrome');
  }
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

    if (params.mode == 'raytracing') {
      let gsd = new GSplatsDataUniformStruct(params);
      let u = raytracingPass.material.uniforms;
      u.bvh.value.updateFrom(bvh);
      u.frameId.value = frameId;
      u.cameraWorldMatrix.value.copy(camera.matrixWorld);
      u.projectionMatrix.value.copy(camera.projectionMatrix);
      u.modelWorldMatrix.value.copy(pointCloud.matrixWorld);
      u.pixelData.value = pixelsRT.rtA.texture;
      u.gsd.value = gsd;
      renderer.setRenderTarget(pixelsRT.rtB);
      raytracingPass.render(renderer);
      pixelsRT.swap();
    }

    if (params.mode == 'raymarching') {
      runRaymarchingPass(renderer, camera, pointCloud, params, frameId, pixelsRT);
    }

    runRenderPass(renderer, params, frameId, pixelsRT.rtA);
    frameId++;
  }
}

init();
render();
