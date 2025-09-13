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

  mode: 'points',
  strategy: SAH,
  maxLeafTris: 8,
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

        void main() {

          vec2 ndc = vUv*2. - 1.;
          vec3 rayOrigin, rayDirection;
          ndcToCameraRay(
            ndc, inverse(modelMatrix) * cameraWorldMatrix, inverse(projectionMatrix),
            rayOrigin, rayDirection);
          rayDirection = normalize(rayDirection);

          gl_FragColor = vec4(0);

          for (int step = 0; step < MAX_RAYCASTS; step++) {
            BVHIntersectResult res;
            bvhIntersectSplats( bvh, rayOrigin, rayDirection, splatSize, res );

            #if SHOW_COST
              
              vec4 cost = vec4(res.numLookupsBVH, res.numLookupsSplats, res.count, 0);
              //cost *= vec4(1,0,0,0);
              gl_FragColor += mat4x4(9,3,1,1, 3,1,9,1, 1,9,3,1, 9,9,9,1) * clamp(cost/1e3, 0., 1.);

            #else

              // blend all splats along the ray
              for (int i = 0; i < res.count; i++) {
                vec3 pos = texelFetch1D( bvh.position, res.splatId[i] ).xyz;
                vec3 dir = rayOrigin + rayDirection*res.dist[i] - pos;
                float x = length(dir)/splatSize*3.5;
                float gaussian = splatOpacity*exp(-x*x*0.5);
                vec4 color = vec4(4, 2, 1, gaussian);
                //float arc = 2.0*sqrt(splatSize*splatSize - dot(dir, dir));
                //vec3 color = vec3(1) * max(0., dot(reflect(rayDirection, normalize(dir)), vec3(0,1,0)));
                
                color.rgb *= color.a;
                gl_FragColor += color * (1. - gl_FragColor.a);
              }

            #endif
            
            if (res.count < MAX_SPLATS_PER_RAY) break;
            rayOrigin += rayDirection*res.dist[res.count - 1];
          }

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

  const bvhGeometry = new THREE.BufferGeometry();
  const position = geometry.attributes.position;
  const index = [];
  for (let i = 0; i < position.count; i++)
    if (i % 1 == 0) index.push(i, i, i);
  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position);
  bvhGeometry.computeBoundsTree(getBVHOptions());
  console.log('Geometry size:', index.length / 3, 'points');

  bvhMesh = new THREE.Mesh(bvhGeometry, new THREE.MeshBasicMaterial({ color: 0xff0000 }));
  bvhHelper = new MeshBVHHelper(bvhMesh, params.depth);
  bvhHelper.name = 'BVH Helper';
  scene.add(bvhHelper);

  updateBVH();
}

function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();

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
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'raytracing']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raytracing') {
    displayFolder.add(params, 'maxRaycasts', 1, 16, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'maxSplatsPerRay', 1, 16, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'splatOpacity', 0, 1.5, 0.01);
    displayFolder.add(params, 'splatSize', 0, 0.05, 0.001);
    displayFolder.add(params, 'showCost').onChange(() => {
      raytracingPass.material.updateDefines();
    });
  }

}

function updateBVH() {
  console.time('MeshBVH');
  bvh = new MeshBVH(bvhMesh.geometry, getBVHOptions());
  console.timeEnd('MeshBVH');
  rebuildGUI();
}

function render() {

  stats.update();
  requestAnimationFrame(render);

  if (params.mode === 'points') {

    if (!pointCloud) return;
    pointCloud.material.size = params.splatSize;
    renderer.render(scene, camera);

  } else if (params.mode === 'raytracing') {
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
