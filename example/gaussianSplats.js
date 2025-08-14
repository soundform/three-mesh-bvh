import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import Stats from 'stats.js';
import { RenderSDFLayerMaterial } from './utils/RenderSDFLayerMaterial.js';
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
  regenerate: () => updateSDF(),

  resolution: 75,
  margin: 0.2,
  mode: 'raymarching',
  surface: 0.01,
  maxSteps: 100,
  displayHelper: true,
  helperDepth: 10,
  displayParents: false,
  strategy: SAH,
  pointSize: 0.005,
};

let renderer, camera, scene, gui, stats;
let outputContainer, bvh, rtSDF;
let bvhMesh, bvhHelper, pointCloud;
let generateSdfPass, copyPass, layerPass, raymarchPass;
const inverseBoundsMatrix = new THREE.Matrix4();
const plyPath = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
const glbPath = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';

class GenerateSDFMaterial2 extends THREE.ShaderMaterial {

  constructor(params) {

    super({

      uniforms: {

        inverseBoundsMatrix: { value: new THREE.Matrix4() },
        zValue: { value: 0 },
        bvh: { value: new MeshBVHUniformStruct() }

      },

      vertexShader: /* glsl */`

        varying vec2 vUv;

        void main() {

          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

        }

      `,

      fragmentShader: /* glsl */`

        precision highp isampler2D;
        precision highp usampler2D;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        ${BVHShaderGLSL.bvh_raymarch_functions}

        varying vec2 vUv;

        uniform BVH bvh;
        uniform float zValue;
        uniform mat4 inverseBoundsMatrix;

        void main() {
          vec3 uvz = vec3( vUv, zValue );
          vec3 point = ( inverse(inverseBoundsMatrix) * vec4(uvz - 0.5, 1) ).xyz;
          float dist = bvhClosestPointToPoint( bvh, point, 1e6 );
          gl_FragColor = vec4( dist, 0, 0, 0 );

        }

      `

    });

    this.setValues(params);

  }

}

class RayMarchSDFMaterial2 extends THREE.ShaderMaterial {

  constructor(params) {

    super({

      defines: {

        MAX_STEPS: 500,
        SURFACE_EPSILON: 0.001,

      },

      uniforms: {

        surface: { value: 0 },
        bvh: { value: new MeshBVHUniformStruct() },
        sdfTex: { value: null },
        inverseBoundsMatrix: { value: new THREE.Matrix4() },
        projectionInverse: { value: new THREE.Matrix4() },
        sdfTransformInverse: { value: new THREE.Matrix4() }

      },

      vertexShader: /* glsl */`

        varying vec2 vUv;

        void main() {

          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

        }

      `,

      fragmentShader: /* glsl */`
        precision highp sampler3D;

        varying vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        ${BVHShaderGLSL.bvh_raymarch_functions}

        uniform float surface;
        uniform BVH bvh;
        uniform sampler3D sdfTex;
        uniform mat4 projectionInverse;
        uniform mat4 sdfTransformInverse;
        uniform mat4 inverseBoundsMatrix;

        #include <common>

        // distance to box bounds
        vec2 rayBoxDist( vec3 boundsMin, vec3 boundsMax, vec3 rayOrigin, vec3 rayDir ) {

          vec3 t0 = ( boundsMin - rayOrigin ) / rayDir;
          vec3 t1 = ( boundsMax - rayOrigin ) / rayDir;
          vec3 tmin = min( t0, t1 );
          vec3 tmax = max( t0, t1 );

          float distA = max( max( tmin.x, tmin.y ), tmin.z );
          float distB = min( tmax.x, min( tmax.y, tmax.z ) );

          float distToBox = max( 0.0, distA );
          float distInsideBox = max( 0.0, distB - distToBox );
          return vec2( distToBox, distInsideBox );

        }

        void main() {

          // get the inverse of the sdf box transform
          mat4 sdfTransform = inverse( sdfTransformInverse );

          // convert the uv to clip space for ray transformation
          vec2 clipSpace = 2.0 * vUv - vec2( 1.0 );

          // get world ray direction
          vec3 rayOrigin = vec3( 0.0 );
          vec4 homogenousDirection = projectionInverse * vec4( clipSpace, - 1.0, 1.0 );
          vec3 rayDirection = normalize( homogenousDirection.xyz / homogenousDirection.w );

          // transform ray into local coordinates of sdf bounds
          vec3 sdfRayOrigin = ( sdfTransformInverse * vec4( rayOrigin, 1.0 ) ).xyz;
          vec3 sdfRayDirection = normalize( ( sdfTransformInverse * vec4( rayDirection, 0.0 ) ).xyz );

          // find whether our ray hits the box bounds in the local box space
          vec2 boxIntersectionInfo = rayBoxDist( vec3( - 0.5 ), vec3( 0.5 ), sdfRayOrigin, sdfRayDirection );
          float distToBox = boxIntersectionInfo.x;
          float distInsideBox = boxIntersectionInfo.y;
          vec3 sdfDims = vec3(textureSize(sdfTex, 0));
          float cost = 0.;

          gl_FragColor = vec4( 0.0 );

          if ( distInsideBox > 0.0 ) {

            // find the surface point in world space
            bool intersectsSurface = false;
            vec4 localPoint = vec4( sdfRayOrigin + sdfRayDirection * ( distToBox + 1e-5 ), 1.0 );
            vec4 point = sdfTransform * localPoint;

            // ray march
            for ( int i = 0; i < MAX_STEPS; i ++ ) {

              cost += 0.01;

              // sdf box extends from - 0.5 to 0.5
              // transform into the local bounds space [ 0, 1 ] and check if we're inside the bounds
              vec3 uv = ( sdfTransformInverse * point ).xyz + 0.5;
              if (clamp(uv, 0., 1.) != uv) break;

              vec4 pt3d = inverse(inverseBoundsMatrix)*vec4(uv - 0.5, 1);
              vec4 pt3d2 = inverse(inverseBoundsMatrix)*vec4(uv - 0.5 + 1.0/sdfDims, 1);
              float distRange = length(pt3d - pt3d2);
              
              //float approxDist = texture(sdfTex, uv).x;
              float approxDist = texelFetch(sdfTex, ivec3(uv*sdfDims), 0).x;
              
              //float maxDist = approxDist + distRange*0.5;
              //float exactDist = bvhClosestPointToPoint(bvh, pt3d.xyz, maxDist);
              //if (abs(approxDist - exactDist) > distRange*0.5) {
              //  gl_FragColor = vec4(0,1,0,1);
              //  return;
              //}

              float dist = approxDist - surface;

              if ( dist < SURFACE_EPSILON ) {
                intersectsSurface = true;
                break;
              }

              point.xyz += rayDirection * abs(dist);
            }

            if ( intersectsSurface ) {
              gl_FragColor.a = 1.0;
              gl_FragColor.rgb = cost*vec3(9,3,1);
            }

          }

          //#include <colorspace_fragment>

        }
      `

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

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(1, 1, 1);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xffffff, 0.2));

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50);
  camera.position.set(1, 1, 2);
  camera.far = 100;
  camera.updateProjectionMatrix();

  new OrbitControls(camera, renderer.domElement);

  stats = new Stats();
  document.body.appendChild(stats.dom);

  // sdf pass to generate the 3d texture
  generateSdfPass = new FullScreenQuad(new GenerateSDFMaterial2());

  // screen pass to render a single layer of the 3d texture
  layerPass = new FullScreenQuad(new RenderSDFLayerMaterial());
  layerPass.material.defines.DISPLAY_GRID = 1;
  layerPass.material.needsUpdate = true;

  // screen pass to render the sdf ray marching
  raymarchPass = new FullScreenQuad(new RayMarchSDFMaterial2());
  raymarchPass.material.defines.MAX_STEPS = params.maxSteps;
  raymarchPass.material.needsUpdate = true;

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
  //let geometry = await new PLYLoader().loadAsync(plyPath);
  let gltf = await new GLTFLoader()
    .setMeshoptDecoder(MeshoptDecoder)
    .loadAsync(glbPath);
  gltf.scene.updateMatrixWorld(true);
  let gltfMesh = gltf.scene.children[0];
  let geometry = gltfMesh.geometry;
  geometry.center();
  console.timeEnd('loadGeometry');
  return geometry;
}

async function initGeometry() {
  const geometry = await loadGeometry();
  const material = new THREE.PointsMaterial({ size: params.pointSize, vertexColors: true });
  pointCloud = new THREE.Points(geometry, material);
  pointCloud.name = 'Point Cloud';
  pointCloud.matrixAutoUpdate = false;
  scene.add(pointCloud);

  const bvhGeometry = new THREE.BufferGeometry();
  const position = geometry.attributes.position;
  const index = [];
  for (let i = 0; i < position.count; i++)
    index.push(i, i, i);
  bvhGeometry.setIndex(index);
  bvhGeometry.setAttribute('position', position);
  bvhGeometry.computeBoundsTree({ strategy: params.strategy });
  console.log('Geometry size:', index.length/3, 'points');

  bvhMesh = new THREE.Mesh(bvhGeometry, new THREE.MeshBasicMaterial({ color: 0xff0000 }));
  bvhHelper = new MeshBVHHelper(bvhMesh, params.depth);
  bvhHelper.name = 'BVH Helper';
  scene.add(bvhHelper);

  updateSDF();
}

// build the gui with parameters based on the selected display mode
function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();

  const generationFolder = gui.addFolder('generation');
  generationFolder.add(params, 'resolution', 10, 200, 1);
  generationFolder.add(params, 'margin', 0, 1);
  generationFolder.add(params, 'regenerate');

  const helperFolder = gui.addFolder('helper');
  helperFolder.add(params, 'displayHelper').onChange(v => {
    bvhHelper.visible = v;
  });
  helperFolder.add(params, 'displayParents').onChange(v => {
    bvhHelper.displayParents = v;
    bvhHelper.update();
  });
  helperFolder.add(params, 'helperDepth', 1, 20, 1).name('depth').onChange(v => {
    bvhHelper.depth = parseInt(v);
    bvhHelper.update();
  });
  helperFolder.open();

  const pointsFolder = gui.addFolder('points');
  pointsFolder.add(params, 'strategy', { CENTER, AVERAGE, SAH }).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree({ strategy: parseInt(v) });
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateSDF();
  });

  pointsFolder.add(params, 'pointSize', 0.001, 0.01, 0.001).onChange(v => {
    pointCloud.material.size = v;
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['geometry', 'raymarching', 'grid layers']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raymarching') {

    displayFolder.add(params, 'surface', 0, 0.1);
    displayFolder.add(params, 'maxSteps', 0, 500, 1).onChange(v => {
      raymarchPass.material.defines.MAX_STEPS = parseInt(v);
      raymarchPass.material.needsUpdate = true;
    });
  }

}

// update the sdf texture based on the selected parameters
function updateSDF() {
  console.time('MeshBVH');
  bvh = new MeshBVH(bvhMesh.geometry, { strategy: params.strategy });
  console.timeEnd('MeshBVH');

  const size3d = params.resolution;
  const matrix = new THREE.Matrix4();
  const center = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3();

  // compute the bounding box of the geometry including the margin which is used to
  // define the range of the SDF
  const bbox = bvh.geometry.boundingBox;
  bbox.getCenter(center);
  scale.subVectors(bbox.max, bbox.min);
  scale.x += 2 * params.margin;
  scale.y += 2 * params.margin;
  scale.z += 2 * params.margin;
  matrix.compose(center, quat, scale);
  inverseBoundsMatrix.copy(matrix).invert();

  const pxWidth = 1 / size3d;
  const halfWidth = 0.5 * pxWidth;
  const startTime = window.performance.now();
  const oesFloatLinear = renderer.extensions.get('OES_texture_float_linear');

  rtSDF?.dispose();
  rtSDF = new THREE.WebGL3DRenderTarget(size3d, size3d, size3d);
  rtSDF.texture.format = THREE.RedFormat;
  rtSDF.texture.type = oesFloatLinear ? THREE.FloatType : THREE.HalfFloatType;
  rtSDF.texture.minFilter = THREE.LinearFilter;
  rtSDF.texture.magFilter = THREE.LinearFilter;
  renderer.initRenderTarget(rtSDF);

  generateSdfPass.material.uniforms.bvh.value.updateFrom(bvh);
  generateSdfPass.material.uniforms.inverseBoundsMatrix.value.copy(inverseBoundsMatrix);

  const scratchTarget = new THREE.WebGLRenderTarget(size3d, size3d);
  scratchTarget.texture.format = THREE.RedFormat;
  scratchTarget.texture.type = oesFloatLinear ? THREE.FloatType : THREE.HalfFloatType;

  // render into each layer
  console.time('updateSDF');

  for (let i = 0; i < size3d; i++) {

    generateSdfPass.material.uniforms.zValue.value = i * pxWidth + halfWidth;
    renderer.setRenderTarget(scratchTarget);
    generateSdfPass.render(renderer);

    // copy the data into the 3d texture since rendering directly
    // into the target causes significant gpu artifacts
    renderer.copyTextureToTexture(
      scratchTarget.texture, rtSDF.texture,
      null, new THREE.Vector3(0, 0, i));
  }

  // initiate read back to get a rough estimate of time taken to generate the sdf
  renderer.copyTextureToTexture(rtSDF.texture, scratchTarget.texture);
  syncTexture2D(scratchTarget.texture);

  renderer.setRenderTarget(null);
  scratchTarget.dispose();
  console.timeEnd('updateSDF');

  const delta = window.performance.now() - startTime;
  outputContainer.innerText = `${delta.toFixed(0)}ms`;

  rebuildGUI();
}

function syncTexture2D(texture) {
  // readRenderTargetPixels works only with RGBA textures
  if (!copyPass) {
    const copyShaderMaterial = new THREE.ShaderMaterial(CopyShader);
    copyPass = new FullScreenQuad(copyShaderMaterial);
  }
  const rtDummy = new THREE.WebGLRenderTarget; // 1 x 1 x RGBA x uint8
  renderer.setRenderTarget(rtDummy);
  copyPass.material.uniforms.tDiffuse.value = texture;
  copyPass.render(renderer);
  renderer.readRenderTargetPixels(rtDummy, 0, 0, 1, 1, new Uint8Array(4));
}

function render() {

  stats.update();
  requestAnimationFrame(render);

  if (params.mode === 'geometry') {

    // render the rasterized geometry
    renderer.render(scene, camera);

  } else if (params.mode === 'grid layers') {
    if (!rtSDF)
      return; // not ready

    const material = layerPass.material;
    material.uniforms.layer.value = 0;
    material.uniforms.sdfTex.value = rtSDF.texture;
    material.uniforms.layers.value = rtSDF.texture.image.depth;
    layerPass.render(renderer);

  } else if (params.mode === 'raymarching') {
    if (!rtSDF)
      return; // not ready

    // render the ray marched texture
    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    raymarchPass.material.uniforms.bvh.value.updateFrom(bvh);
    raymarchPass.material.uniforms.sdfTex.value = rtSDF.texture;
    raymarchPass.material.uniforms.surface.value = params.surface;
    raymarchPass.material.uniforms.inverseBoundsMatrix.value.copy(inverseBoundsMatrix);
    raymarchPass.material.uniforms.projectionInverse.value.copy(camera.projectionMatrixInverse);
    raymarchPass.material.uniforms.sdfTransformInverse.value.copy(pointCloud.matrixWorld)
      .invert().premultiply(inverseBoundsMatrix).multiply(camera.matrixWorld);
    raymarchPass.render(renderer);

  }
}

init();
render();
