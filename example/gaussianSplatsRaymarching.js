import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';
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
  regenerate: () => updateSDF(),

  resolution: 80,
  mode: 'points',
  maxSteps: 100,
  strategy: SAH,
  maxLeafTris: 8,
  pointSize: 0.01,
  showCost: true,
};

const getBVHOptions = () => ({ strategy: params.strategy, maxLeafTris: params.maxLeafTris });

let renderer, camera, scene, gui, stats;
let outputContainer, bvh, rtSDF, rtJFA, rtSDFandJFA;
let bvhMesh, bvhHelper, pointCloud;
let sdfVoxelsPass, copyPass, previewPass, jfaPass, combinePass, raymarchPass;
const boundsMatrix = new THREE.Matrix4();
//const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/point-cloud-porsche/scene.ply';
const sceneFile = 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/stanford-bunny/bunny.glb';

class SDFVoxelsPass extends THREE.ShaderMaterial {

  constructor(params) {

    super({

      uniforms: {

        boundsMatrix: { value: new THREE.Matrix4() },
        zValue: { value: 0 },
        pointSize: { value: 0 },
        sdfDims: { value: new THREE.Vector3() },
        bvh: { value: new MeshBVHUniformStruct() }

      },

      vertexShader: /* glsl */`

        varying vec2 vUv;

        void main() {

          vUv = uv;
          gl_Position = vec4( position, 1.0 );

        }

      `,

      fragmentShader: /* glsl */`

        precision highp isampler2D;
        precision highp usampler2D;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        ${BVHShaderGLSL.bvh_raymarching_functions}

        varying vec2 vUv;

        uniform BVH bvh;
        uniform vec3 sdfDims;
        uniform float zValue;
        uniform float pointSize;
        uniform mat4 boundsMatrix;

        void main() {
          vec3 uv = vec3( vUv, zValue );
          vec3 point = ( boundsMatrix * vec4(uv - 0.5, 1) ).xyz;
          vec3 margin = (boundsMatrix * vec4(vec3(0.5)/sdfDims, 0)).xyz;

          // Find all points that overlap with this voxel.
          PointsRange pr = bvhClosestPointToPoint( bvh, pointSize, point, margin, 0., 0u );

          gl_FragColor = vec4( pr.commonNode, 0, 0, 0 );

          // Mark empty voxels.
          if (pr.countPts == 0)
            gl_FragColor.x = -1.;
        }

      `

    });

    this.setValues(params);

  }

}

class VoxelsPreviewPass extends THREE.ShaderMaterial {
  constructor(params) {
    super({
      uniforms: {
        iVoxels: { value: null },
      },

      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        precision highp sampler3D;
        varying vec2 vUv;
        uniform sampler3D iVoxels;

        void main() {
          ivec3 size = textureSize(iVoxels, 0);
          float dim = ceil( sqrt( float(size.z) ) );
          vec2 cell = floor( vUv * dim );
          vec2 frac = vUv * dim - cell;
          float zLayer = ( cell.y * dim + cell.x ) / ( dim * dim );
          vec3 uvz = vec3( frac, zLayer );
          vec4 vox = texture( iVoxels, uvz );
          
          //float temp = vox.x; // distance
          //float temp = vox.y / 1e3; // common node id

          int nodeId = int(vox.x);
          ivec3 p = (ivec3(-nodeId-1) >> ivec3(0,8,16)) & 255;
          vec3 uvz0 = (0.5 + vec3(p)) / vec3(size);

          //gl_FragColor = vec4(uvz0,1);
          //return;

          float temp = length(uvz - uvz0);
          if (nodeId >= 0) temp = 0.; // non-empty voxel

          //gl_FragColor.rgb = vox.xyz * 255. / vec3(size);
          gl_FragColor.rgb = temp * vec3(9,3,1);
          gl_FragColor.a = 1.0;
        }
      `
    });

    this.setValues(params);
  }
}

// JFA[p] = the nearest non-empty SDF voxel.
class JumpFloodingPass extends THREE.ShaderMaterial {
  constructor(params) {
    super({
      uniforms: {
        iJFA: { value: null },
        iSDF: { value: null },
        zValue: { value: 0 },
        iStep: { value: 0 },
        boundsMatrix: { value: new THREE.Matrix4() },
      },

      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1);
        }
      `,

      fragmentShader: /* glsl */`
        precision highp sampler3D;

        varying vec2 vUv;

        uniform sampler3D iSDF;
        uniform sampler3D iJFA;
        uniform float zValue;
        uniform int iStep;
        uniform mat4 boundsMatrix;

        #define INF 1e6
        #define JFA(p) ivec3(texelFetch(iJFA, p, 0).xyz*255.) // reads from a uint8 texture
        #define SDF(p) (texelFetch(iSDF, p, 0).x >= 0.) // detect non-empty voxels
        #define POS(uvz) (boundsMatrix*vec4((uvz) - 0.5, 1)).xyz // uvz -> world coords
        #define DIST(p) distance(pos, POS(vec3(p)/vec3(r))) // JFA metric: proximity to uvz
        #define MAX3(a,b,c) max(max(a,b),c)

        void main() {
          vec3 uvz = vec3(vUv, zValue);
          vec3 pos = POS(uvz);
          ivec3 r = textureSize(iSDF, 0);
          ivec3 p = ivec3(uvz * vec3(r));
          int n = int(log2(float(MAX3(r.x, r.y, r.z))));
          int f = iStep % n;
          float jump = exp2(float(n - 1 - f)); // 1, 2, 3, ... -> 2048, 1024, 512, ...

          ivec3 o = iStep == 0 ? p : JFA(p);
          float d = SDF(o) ? DIST(o) : INF;
          
          for (int x = -1; x <= 1; x++)
          for (int y = -1; y <= 1; y++)
          for (int z = -1; z <= 1; z++)
          {
              if (vec3(x,y,z) == vec3(0))
                continue;
              
              ivec3 p2 = ivec3(mod(vec3(p) + jump*vec3(x,y,z), vec3(r)));
              ivec3 o2 = iStep == 0 ? p2 : JFA(p2);
              float d2 = SDF(o2) ? DIST(o2) : INF;

              if (d2 < d) d = d2, o = o2;
          }

          gl_FragColor = vec4(o, 0)/255.; // renders to a uint8 texture
        }
      `
    });

    this.setValues(params);
  }
}

// Merge SDF & JFA textures to do 1 lookup instead of 2.
class CombinePass extends THREE.ShaderMaterial {
  constructor(params) {
    super({
      uniforms: {
        iSDF: { value: null },
        iJFA: { value: null },
        zValue: { value: 0 },
      },

      vertexShader: /* glsl */`
        out vec2 vUv;
        
        void main() {
          vUv = uv;
          gl_Position = vec4( position, 1.0 );
        }
      `,

      fragmentShader: /* glsl */`
        precision highp sampler3D;
        
        in vec2 vUv;

        uniform sampler3D iSDF;
        uniform sampler3D iJFA;
        uniform float zValue;

        #define T(tex, uvz) texelFetch(tex, ivec3(uvz * vec3(textureSize(tex, 0))), 0)

        void main() {
          vec3 uvz = vec3(vUv, zValue);
          int nodeId = int(T(iSDF, uvz).x);
          ivec3 vox = ivec3(T(iJFA, uvz).xyz*255.); // reads from uint8 texture
          // nodeId = 0 is possible, and vox = 0 is also possible
          if (nodeId < 0) nodeId = -1 - (vox.x + (vox.y << 8) + (vox.z << 16));
          gl_FragColor.x = float(nodeId);
        }
      `
    });

    this.setValues(params);
  }
}

class RaymarchingPass extends THREE.ShaderMaterial {

  constructor(params) {

    super({

      defines: {

        MAX_STEPS: 500,

      },

      uniforms: {

        showCost: { value: true },
        pointSize: { value: 0 },
        bvh: { value: new MeshBVHUniformStruct() },
        iSDFandJFA: { value: null },
        boundsMatrix: { value: new THREE.Matrix4() },
        projectionInverse: { value: new THREE.Matrix4() },
        sdfTransformInverse: { value: new THREE.Matrix4() }

      },

      vertexShader: /* glsl */`

        varying vec2 vUv;

        void main() {

          vUv = uv;
          gl_Position = vec4( position, 1.0 );

        }

      `,

      fragmentShader: /* glsl */`
        precision highp sampler3D;

        varying vec2 vUv;

        ${BVHShaderGLSL.common_functions}
        ${BVHShaderGLSL.bvh_struct_definitions}
        ${BVHShaderGLSL.bvh_raymarching_functions}

        uniform bool showCost;
        uniform float pointSize;
        uniform BVH bvh;
        uniform sampler3D iSDFandJFA;
        uniform mat4 projectionInverse;
        uniform mat4 sdfTransformInverse;
        uniform mat4 boundsMatrix;

        #define SDF(p) int(texelFetch(iSDFandJFA, ivec3(p), 0).x)
        #define JFA(p) ((ivec3(-p-1) >> ivec3(0,8,16)) & 255)
        #define POS(uvz) (boundsMatrix * vec4(vec3(uvz) - 0.5, 1)).xyz
        #define DCHECK(x) if (!(x)) { gl_FragColor = vec4(0,1,0,1); return; }

        #include <common>

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

        float sdBox(vec3 p, vec3 h) {
          return length(p - clamp(p, -h, h));
        }

        vec3 hash33(vec3 p3) {
          p3 = fract(p3 * vec3(.1031, .1030, .0973));
          p3 += dot(p3, p3.yxz+33.33);
          return fract((p3.xxy + p3.yxx)*p3.zyx);
        }

        void main() {

          // get the inverse of the sdf box transform
          mat4 sdfTransform = inverse( sdfTransformInverse );

          // get world ray direction
          vec3 rayOrigin = vec3( 0.0 );
          vec4 homogenousDirection = projectionInverse * vec4(vUv*2. - 1., -1, 1);
          vec3 rayDirection = normalize( homogenousDirection.xyz / homogenousDirection.w );
          vec3 raydir3 = normalize(POS(rayDirection) - POS(0));

          vec3 sdfRayOrigin = ( sdfTransformInverse * vec4( rayOrigin, 1.0 ) ).xyz;
          vec3 sdfRayDirection = normalize( ( sdfTransformInverse * vec4( rayDirection, 0.0 ) ).xyz );

          // find whether our ray hits the box bounds in the local box space
          vec2 boxIntersectionInfo = rayBoxDist( vec3(-0.5), vec3(0.5), sdfRayOrigin, sdfRayDirection );
          float distToBox = boxIntersectionInfo.x + 1e-5;
          float distInsideBox = boxIntersectionInfo.y;
          vec3 sdfDims = vec3(textureSize(iSDFandJFA, 0));
          vec3 margin = POS(vec3(0.5)/sdfDims) - POS(0);
          vec4 cost = vec4(0);
          vec3 normal = vec3(0);
          vec3 color = vec3(0);

          gl_FragColor = vec4(0);

          if ( distInsideBox <= 0.0 )
            return;

          // find the surface point in world space
          vec4 point = sdfTransform * vec4( sdfRayOrigin + sdfRayDirection * distToBox, 1.0 );
          float minStep = pointSize*0.1;

          for ( int i = 0; i < MAX_STEPS; i ++ ) {

            // sdf box extends from - 0.5 to 0.5
            // transform into the local bounds space [ 0, 1 ] and check if we're inside the bounds
            vec3 uvz = ( sdfTransformInverse * point ).xyz + 0.5;
            vec3 point3 = POS(uvz);
            if (clamp(uvz, 0., 1.) != uvz) break;

            ivec3 tex3 = ivec3(uvz*sdfDims);
            float diff = length(point3 - POS((0.5 + vec3(tex3))/sdfDims)); // distance to the voxel center
            cost.z++; // green
            int nodeId = SDF(tex3);
            float dist = 0.;
            
            if (nodeId >= 0) {
              // If the voxel is not empty, find the exact SDF value.
              float maxDist = length(margin) + diff; // it should be distance to the furthest corner of the voxel
              PointsRange pr = bvhClosestPointToPoint(bvh, pointSize, point3.xyz, vec3(0), maxDist, uint(nodeId));

              cost.x += float(pr.lookupsPts); // orange
              cost.y += float(pr.lookupsBVH); // purple
              cost.w += float(pr.countPts);

              dist = pr.dist;

              if ( dist < minStep ) {
                normal = normalize(point3.xyz - pr.closestPoint);
                color = hash33(pr.closestPoint);
                break;
              }
            } else {
              // If the voxel is empty, jump to the nearest non-empty voxel.
              ivec3 vox = JFA(nodeId);
              DCHECK(vox == clamp(vox, ivec3(0), ivec3(sdfDims) - 1));
              vec3 voxPos = POS((0.5 + vec3(vox))/sdfDims);
              dist = sdBox(point3 - voxPos, margin);
            }

            point.xyz += rayDirection * max(dist, minStep);
          }

          if (showCost) {
            cost *= 0.001; // * vec4(0,0,0,1);
            gl_FragColor = mat4x4(9,3,1,1, 3,1,9,1, 1,9,3,1, 10,10,10,1) * clamp(cost, 0., 1.);
          } else if (length(normal) > 0.) {
            vec3 shade = color * max(0., dot(reflect(rayDirection, normal), vec3(0,1,0)));
            gl_FragColor.rgb += shade * (1. - gl_FragColor.a);
          }

          gl_FragColor.a = 1.0;
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
  scene.add(new THREE.AmbientLight());

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50);
  camera.position.set(1, 1, 2);
  camera.far = 100;
  camera.updateProjectionMatrix();

  new OrbitControls(camera, renderer.domElement);

  stats = new Stats();
  document.body.appendChild(stats.dom);

  sdfVoxelsPass = new FullScreenQuad(new SDFVoxelsPass());
  previewPass = new FullScreenQuad(new VoxelsPreviewPass());
  jfaPass = new FullScreenQuad(new JumpFloodingPass);
  combinePass = new FullScreenQuad(new CombinePass);

  raymarchPass = new FullScreenQuad(new RaymarchingPass());
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
  const material = new THREE.PointsMaterial({ size: params.pointSize, vertexColors: true });
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

  updateSDF();
}

// build the gui with parameters based on the selected display mode
function rebuildGUI() {
  gui?.destroy();
  gui = new GUI();

  const generationFolder = gui.addFolder('generation');
  generationFolder.add(params, 'resolution', 10, 200, 10);
  generationFolder.add(params, 'regenerate');

  const pointsFolder = gui.addFolder('points');
  pointsFolder.add(params, 'strategy', { CENTER, AVERAGE, SAH }).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree(getBVHOptions());
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateSDF();
  });
  pointsFolder.add(params, 'maxLeafTris', 1, 16, 1).onChange(v => {
    console.time('computeBoundsTree');
    bvh.geometry.computeBoundsTree(getBVHOptions());
    console.timeEnd('computeBoundsTree');
    bvhHelper.update();
    updateSDF();
  });

  pointsFolder.add(params, 'pointSize', 0.001, 0.1, 0.001).onChange(v => {
    pointCloud.material.size = v;
  });
  pointsFolder.open();

  const displayFolder = gui.addFolder('display');
  displayFolder.add(params, 'mode', ['points', 'raymarching', 'SDF voxels']).onChange(v => {
    rebuildGUI();
  });

  if (params.mode === 'raymarching') {
    displayFolder.add(params, 'showCost');
    displayFolder.add(params, 'maxSteps', 0, 500, 1).onChange(v => {
      raymarchPass.material.defines.MAX_STEPS = parseInt(v);
      raymarchPass.material.needsUpdate = true;
    });
  }

}

// update the sdf texture based on the selected parameters
function updateSDF() {
  console.time('MeshBVH');
  bvh = new MeshBVH(bvhMesh.geometry, getBVHOptions());
  console.timeEnd('MeshBVH');

  const size3d = params.resolution;
  const center = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3();

  // compute the bounding box of the geometry including the margin which is used to
  // define the range of the SDF
  const bbox = bvh.geometry.boundingBox;
  bbox.getCenter(center);
  scale.subVectors(bbox.max, bbox.min);
  scale.x += 2 * params.pointSize;
  scale.y += 2 * params.pointSize;
  scale.z += 2 * params.pointSize;
  boundsMatrix.compose(center, quat, scale);
  console.debug('boundsMatrix:', boundsMatrix.elements.join(', '));

  const startTime = window.performance.now();

  // update SDF voxels

  rtSDF?.dispose();
  rtSDF = new THREE.WebGL3DRenderTarget(size3d, size3d, size3d);
  rtSDF.texture.format = THREE.RedFormat;
  rtSDF.texture.type = THREE.FloatType;

  sdfVoxelsPass.material.uniforms.pointSize.value = params.pointSize;
  sdfVoxelsPass.material.uniforms.sdfDims.value.set(size3d, size3d, size3d);
  sdfVoxelsPass.material.uniforms.bvh.value.updateFrom(bvh);
  sdfVoxelsPass.material.uniforms.boundsMatrix.value.copy(boundsMatrix);

  renderTexture3D(sdfVoxelsPass, rtSDF, 'SDF');

  // update JFA voxels

  rtJFA?.dispose();
  rtJFA = new THREE.WebGL3DRenderTarget(size3d, size3d, size3d); // RGBA x uint8
  let rtTMP = rtJFA.clone();

  jfaPass.material.uniforms.iSDF.value = rtSDF.texture;
  jfaPass.material.uniforms.boundsMatrix.value.copy(boundsMatrix);

  for (let i = 0; i < Math.floor(Math.log2(size3d)); i++) {
    jfaPass.material.uniforms.iJFA.value = rtJFA.texture;
    jfaPass.material.uniforms.iStep.value = i;
    renderTexture3D(jfaPass, rtTMP, 'JFA.' + i);
    [rtJFA, rtTMP] = [rtTMP, rtJFA];
  }

  rtTMP.dispose();

  // combine SDF and JFA textures

  rtSDFandJFA?.dispose();
  rtSDFandJFA = new THREE.WebGL3DRenderTarget(size3d, size3d, size3d);
  rtSDFandJFA.texture.format = THREE.RedFormat;
  rtSDFandJFA.texture.type = THREE.FloatType;

  combinePass.material.uniforms.iSDF.value = rtSDF.texture;
  combinePass.material.uniforms.iJFA.value = rtJFA.texture;

  renderTexture3D(combinePass, rtSDFandJFA, 'SDF+JFA');

  rtSDF.dispose(); // no longer needed
  rtJFA.dispose(); // no longer needed

  const delta = window.performance.now() - startTime;
  outputContainer.innerText = `${delta.toFixed(0)}ms`;

  rebuildGUI();
}

function renderTexture3D(pass, rt3d, label = pass.name) {
  let { width, height, depth } = rt3d.texture.image;

  const rt2d = new THREE.WebGLRenderTarget(width, height);
  rt2d.texture.format = rt3d.texture.format;
  rt2d.texture.type = rt3d.texture.type;

  renderer.initRenderTarget(rt3d);

  // render into each layer
  console.time(label);

  for (let z = 0; z < depth; z++) {
    pass.material.uniforms.zValue.value = (z + 0.5) / depth;
    renderer.setRenderTarget(rt2d);
    pass.render(renderer);

    // copy the data into the 3d texture since rendering directly
    // into the target causes significant gpu artifacts
    renderer.copyTextureToTexture(
      rt2d.texture, rt3d.texture,
      null, new THREE.Vector3(0, 0, z));
  }

  // initiate read back to get a rough estimate of time taken to generate the sdf
  renderer.copyTextureToTexture(rt3d.texture, rt2d.texture);
  syncTexture2D(rt2d.texture);

  renderer.setRenderTarget(null);
  rt2d.dispose();
  console.timeEnd(label);
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

  if (params.mode === 'points') {

    // render the rasterized geometry
    renderer.render(scene, camera);

  } else if (params.mode === 'SDF voxels') {
    if (!rtJFA)
      return; // not ready

    const material = previewPass.material;
    material.uniforms.iVoxels.value = rtSDFandJFA.texture;
    previewPass.render(renderer);

  } else if (params.mode === 'raymarching') {
    if (!rtSDFandJFA)
      return; // not ready

    // render the ray marched texture
    camera.updateMatrixWorld();
    pointCloud.updateMatrixWorld();

    raymarchPass.material.uniforms.bvh.value.updateFrom(bvh);
    raymarchPass.material.uniforms.iSDFandJFA.value = rtSDFandJFA.texture;
    raymarchPass.material.uniforms.pointSize.value = params.pointSize;
    raymarchPass.material.uniforms.showCost.value = params.showCost;
    raymarchPass.material.uniforms.boundsMatrix.value.copy(boundsMatrix);
    raymarchPass.material.uniforms.projectionInverse.value.copy(camera.projectionMatrixInverse);
    raymarchPass.material.uniforms.sdfTransformInverse.value.copy(pointCloud.matrixWorld)
      .invert().premultiply(boundsMatrix.clone().invert()).multiply(camera.matrixWorld);
    raymarchPass.render(renderer);

  }
}

init();
render();
