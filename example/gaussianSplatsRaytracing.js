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
  splatOpacity: -1, // exp2
  maxRaycasts: 1,
  maxSplatsPerRay: 8,
  showCost: false,
};

const getBVHOptions = () => ({ strategy: params.strategy, maxLeafTris: params.maxLeafTris });

let renderer, camera, scene, gui, stats, outputContainer;
let bvh, bvhMesh, bvhHelper, pointCloud;
let raytracingPass, drawPixelsPass;
let renderTargets = [];
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

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        ${BVHShaderGLSL.bvh_gsplat_ray_functions}

        uniform BVH bvh;
        uniform float splatSize;
        uniform float splatOpacity;
        uniform mat4 cameraWorldMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        uniform sampler2D pixelData;

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
            
            vec4 cost = vec4(res.numLookupsBVH, res.numLookupsSplats, res.numSplats, 0)/1e3;

            // rgba.zw are reserved for raytracing metadata
            cost.xy += unpackUnorm2x16(floatBitsToUint(rgba.x));
            cost.zw += unpackUnorm2x16(floatBitsToUint(rgba.y));

            rgba.x = uintBitsToFloat(packUnorm2x16(cost.xy));
            rgba.y = uintBitsToFloat(packUnorm2x16(cost.zw));

          #else

            // blend all splats along the ray
            for (int i = 0; i < res.numSplats; i++) {

              vec3 pos = texelFetch1D( bvh.position, gSplatIds[i] ).xyz;
              vec3 dir = ro + rd*gSplatDists[i] - pos;
              float d = length(dir)/splatSize*3.0;
              float density = splatOpacity*exp(-d*d*0.5);
              vec3 color = vec3(1,0,0); // luminance
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

          vec2 size = vec2(textureSize(pixelData, 0));
          gl_FragColor = texelFetch(pixelData, ivec2(vUv*size), 0);
          
          if (gl_FragColor.z >= INFINITY) return;
          rayOrigin += gl_FragColor.z*rayDirection;

          for (int step = 0; step < MAX_RAYCASTS; step++) {
            float d = raycast(rayOrigin, rayDirection, gl_FragColor);
            // beware of float32 accuracy
            d += max(d/1e6, splatSize/1e4);
            gl_FragColor.z += d;
            if (d >= INFINITY) break;
            rayOrigin += d*rayDirection;
          }
        }`
    });

    this.setValues(params);
  }
}

class DrawPixelsMaterial extends THREE.ShaderMaterial {
  constructor(params) {
    super({
      uniforms: {
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

        uniform bool showCost;
        uniform sampler2D pixelData;

        void main() {
          vec2 size = vec2(textureSize(pixelData, 0));
          vec4 o = texelFetch(pixelData, ivec2(vUv*size), 0);
          
          if (showCost) {
            vec4 cost;
            cost.xy = unpackUnorm2x16(floatBitsToUint(o.x));
            cost.zw = unpackUnorm2x16(floatBitsToUint(o.y));
            o = mat4x4(9,3,1,0, 3,1,9,0, 1,9,3,0, 3,9,1,0) * cost;
            o.w = 1.0;
          } else {
            o = vec4(9,3,1,1)*o.xxxw;
          }

          gl_FragColor = o;
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

  let orbit = new OrbitControls(camera, renderer.domElement);
  orbit.addEventListener('change', resetRenderState);

  for (let i = 0; i < 2; i++)
    renderTargets[i] = new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType });

  stats = new Stats();
  document.body.appendChild(stats.dom);

  drawPixelsPass = new FullScreenQuad(new DrawPixelsMaterial());
  raytracingPass = new FullScreenQuad(new RaytracingMaterial());
  raytracingPass.material.updateDefines();

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

  for (let rt of renderTargets) {
    rt.setSize(w, h);
    renderer.setRenderTarget(rt);
    renderer.clear();
  }

  renderer.setRenderTarget(null);
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
  pointsFolder.add(params, 'sparsity', 0, 12, 1).onChange(v => {
    updateBVHMesh();
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['rasterizer', 'raytracer']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raytracer') {
    //displayFolder.add(params, 'maxRaycasts', 1, 16, 1).onChange(() => {
    //  raytracingPass.material.updateDefines();
    //});
    displayFolder.add(params, 'maxSplatsPerRay', 1, 64, 1).onChange(() => {
      raytracingPass.material.updateDefines();
    });
    displayFolder.add(params, 'splatOpacity', -10, -1, 1);
    displayFolder.add(params, 'splatSize', 0.001, 0.5, 0.001);
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
    uniforms.splatOpacity.value = 2**params.splatOpacity;
    uniforms.cameraWorldMatrix.value.copy(camera.matrixWorld);
    uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
    uniforms.modelMatrix.value.copy(pointCloud.matrixWorld);
    uniforms.pixelData.value = renderTargets[0].texture;
    renderer.setRenderTarget(renderTargets[1]);
    raytracingPass.render(renderer);

    uniforms = drawPixelsPass.material.uniforms;
    uniforms.showCost.value = params.showCost;
    uniforms.pixelData.value = renderTargets[1].texture;
    renderer.setRenderTarget(null);
    drawPixelsPass.render(renderer);

    renderTargets = [renderTargets[1], renderTargets[0]];
  }
}

init();
render();
