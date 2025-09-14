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
  MeshBVHUniformStruct
} from 'three-mesh-bvh';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;

const params = {
  regenerate: () => updateBVH(),

  mode: 'rasterizer',
  strategy: SAH,
  maxLeafTris: 8,
  sparsity: 0,
  splatSize: 0.005,
  splatOpacity: 0.5,
  maxRaycasts: 2,
  maxSplatsPerRay: 8,
  showCost: false,
};

const getBVHOptions = () => ({ strategy: params.strategy, maxLeafTris: params.maxLeafTris });

let renderer, camera, scene, gui, stats, outputContainer;
let bvh, bvhMesh, bvhHelper, pointCloud;
let raytracingPass;
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';
const sceneFile = 'models/bunny.glb';
//const sceneFile = 'models/sportcar.ply';

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
        MAX_RAYCASTS: 1,
        MAX_SPLATS_PER_RAY: 1,

      },

      uniforms: {

        splatSize: { value: 0 },
        splatOpacity: { value: 0 },
        bvh: { value: new MeshBVHUniformStruct() },

        cameraWorldMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        modelMatrix: { value: new THREE.Matrix4() },

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

        uniform BVH bvh;
        uniform float splatSize;
        uniform float splatOpacity;
        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;

        #include <common>

        vec3 hash33(vec3 p3) {
          p3 = fract(p3 * vec3(.1031, .1030, .0973));
          p3 += dot(p3, p3.yxz+33.33);
          return fract((p3.xxy + p3.yxx)*p3.zyx);
        }

        float raycast(vec3 ro, vec3 rd, inout vec4 rgba) {
          BVHIntersectResult res;
          bvhIntersectSplats( bvh, ro, rd, splatSize, res );

          #if SHOW_COST
            
            vec4 cost = vec4(res.numLookupsBVH, res.numLookupsSplats, res.numSplats, 0);
            rgba += mat4x4(9,3,1,1, 3,1,9,1, 1,9,3,1, 9,9,9,1) * clamp(cost/1e3, 0., 1.);
            rgba.a = 1.0;

          #else

            // blend all splats along the ray
            for (int i = 0; i < res.numSplats; i++) {

              vec3 pos = texelFetch1D( bvh.position, gSplatIds[i] ).xyz;
              vec3 dir = ro + rd*gSplatDists[i] - pos;
              float d = length(dir)/splatSize*3.0;
              float density = splatOpacity*exp(-d*d*0.5);
              vec3 color = vec3(9,3,1); // luminance
              rgba += vec4(color, 1) * density * (1. - rgba.a);
              if (rgba.a > 0.995) return INFINITY;
            }

          #endif
          
          if (res.numSplats < MAX_SPLATS_PER_RAY)
            return INFINITY;
          return gSplatDists[res.numSplats - 1];
        }

        void main() {
          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDirection;
          ndcToCameraRay(
            ndc, inverse(modelMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDirection);
          rayDirection = normalize(rayDirection);

          gl_FragColor = vec4(0);

          for (int step = 0; step < MAX_RAYCASTS; step++) {
            float d = raycast(rayOrigin, rayDirection, gl_FragColor);
            if (d == INFINITY) break;
            // beware of float32 accuracy
            d += max(d/1e6, splatSize/1e4);
            rayOrigin += d*rayDirection;
          }

          gl_FragColor.rgb += vec3(0.05)*(1. - gl_FragColor.a);
          gl_FragColor.a = 1.0;
        }`
    });

    this.setValues(params);
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
  scene.add(new THREE.AmbientLight());

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50);
  camera.position.set(1, 1, 2);
  camera.far = 100;
  camera.updateProjectionMatrix();

  new OrbitControls(camera, renderer.domElement);

  stats = new Stats();
  document.body.appendChild(stats.dom);

  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();

  initGeometry();
  rebuildGUI();

  window.addEventListener('resize', function () {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

  }, false);

}

async function loadGeometry() {
  console.time('loadGeometry');
  console.log('Loading scene:', sceneFile);
  let geometry;

  if (sceneFile.endsWith('.ply')) {
    geometry = await new PLYLoader().loadAsync(sceneFile);
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
  const material = new THREE.PointsMaterial({ size: params.splatSize, vertexColors: true });
  pointCloud = new THREE.Points(geometry, material);
  pointCloud.name = 'Point Cloud';
  pointCloud.matrixAutoUpdate = false;
  scene.add(pointCloud);

  updateBVHMesh();
}

function updateBVHMesh() {
  const bvhGeometry = new THREE.BufferGeometry();
  const position = pointCloud.geometry.attributes.position;
  const index = [];
  for (let i = 0; i < position.count; i++)
    if (i % (1 << params.sparsity) == 0) 
      index.push(i, i, i);
  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position);
  bvhGeometry.computeBoundsTree(getBVHOptions());
  console.log('Geometry size:', index.length / 3, 'rasterizer');

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

  const pointsFolder = gui.addFolder('rasterizer');
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
  pointsFolder.add(params, 'sparsity', 0, 16, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['rasterizer', 'raytracer']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raytracer') {
    displayFolder.add(params, 'maxRaycasts', 1, 16, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'maxSplatsPerRay', 1, 64, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'splatOpacity', 0, 1, 0.01);
    displayFolder.add(params, 'splatSize', 0, 0.5, 0.001);
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

function render() {

  stats.update();
  requestAnimationFrame(render);

  if (params.mode === 'rasterizer') {

    if (!pointCloud) return;
    pointCloud.material.size = params.splatSize;
    renderer.render(scene, camera);

  } else if (params.mode === 'raytracer') {
    if (!bvh) return;

    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    let uniforms = raytracingPass.material.uniforms;

    uniforms.bvh.value.updateFrom(bvh);
    uniforms.splatSize.value = params.splatSize;
    uniforms.splatOpacity.value = params.splatOpacity;
    uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
    uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
    uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);

    raytracingPass.render(renderer);

  }
}

init();
render();
