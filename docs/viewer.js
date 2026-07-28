const container = document.getElementById("terrain-viewer");

if (typeof THREE === "undefined") {
  container.innerHTML = "<p class=\"viewer-error\">Three.js could not be loaded.</p>";
  throw new Error("Three.js is required for the project-page viewer.");
}

if (!THREE.GLTFLoader) {
  container.innerHTML = "<p class=\"viewer-error\">Three.js GLTFLoader could not be loaded.</p>";
  throw new Error("THREE.GLTFLoader is required for the DSM mesh viewer.");
}

const models = {
  ours: {
    label: "SeasonStereo full",
    dsm: "./assets/oma084_mesh_ours.glb",
    disparity: "./assets/oma084_disp_ours.glb",
  },
  diachronic: {
    label: "Diachronic Stereo",
    dsm: "./assets/oma084_mesh_diachronic.glb",
    disparity: "./assets/oma084_disp_diachronic.glb",
  },
  monsterpp: {
    label: "MonSter++",
    dsm: "./assets/oma084_mesh_monsterpp.glb",
    disparity: "./assets/oma084_disp_monsterpp.glb",
  },
  lidar: {
    label: "LiDAR-supervised variant",
    dsm: "./assets/oma084_mesh_lidar.glb",
    disparity: "./assets/oma084_disp_lidar.glb",
  },
  lidargt: {
    label: "LiDAR GT",
    dsm: "./assets/oma084_mesh_gt.glb",
    disparity: null,
  },
  pseudogt: {
    label: "Pseudo-GT only",
    dsm: "./assets/oma084_mesh_pseudogt.glb",
    disparity: "./assets/oma084_disp_pseudogt.glb",
  },
  photo: {
    label: "Pseudo-GT + photo",
    dsm: "./assets/oma084_mesh_photo.glb",
    disparity: "./assets/oma084_disp_photo.glb",
  },
};

const products = {
  disparity: "disparity",
  dsm: "DSM",
};

const state = {
  model: "ours",
  product: "disparity",
  layer: "surface",
};

let activeRoot = null;
let hasFramedOnce = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101719);
scene.fog = new THREE.Fog(0x101719, 260, 520);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
if (THREE.sRGBEncoding) {
  renderer.outputEncoding = THREE.sRGBEncoding;
}
container.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 800);
const orbit = {
  theta: 0.72,
  phi: 1.02,
  radius: 260,
  minRadius: 70,
  maxRadius: 560,
  minPhi: 0.18,
  maxPhi: 1.68,
  target: new THREE.Vector3(0, 18, 0),
  panLimit: 140,
  dragging: false,
  panning: false,
  lastX: 0,
  lastY: 0,
};

const hemiLight = new THREE.HemisphereLight(0xd7fff3, 0x17211f, 1.35);
scene.add(hemiLight);

const sun = new THREE.DirectionalLight(0xffefd0, 2.9);
sun.position.set(110, 160, 82);
scene.add(sun);

const fill = new THREE.DirectionalLight(0x86b8ff, 0.55);
fill.position.set(-110, 58, -92);
scene.add(fill);

const grid = new THREE.GridHelper(400, 16, 0x31433e, 0x22302d);
grid.position.y = -0.25;
grid.material.opacity = 0.24;
grid.material.transparent = true;
scene.add(grid);

const loader = new THREE.GLTFLoader();

installPointerControls();
installUi();
resize();
loadModel(state.model);

window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(container);

renderer.setAnimationLoop((time) => {
  const seconds = time * 0.001;
  sun.position.x = 110 + Math.sin(seconds * 0.16) * 22;
  sun.position.z = 82 + Math.cos(seconds * 0.16) * 24;
  if (!orbit.dragging && !orbit.panning) {
    orbit.theta += 0.0003;
  }
  updateCamera();
  renderer.render(scene, camera);
});

function installUi() {
  document.querySelectorAll("[data-model]").forEach((button) => {
    button.addEventListener("click", () => {
      state.model = button.dataset.model;
      setActive("[data-model]", state.model, "model");
      loadModel(state.model);
    });
  });

  document.querySelectorAll("[data-product]").forEach((button) => {
    button.addEventListener("click", () => {
      state.product = button.dataset.product;
      setActive("[data-product]", state.product, "product");
      loadModel(state.model);
    });
  });

  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      state.layer = button.dataset.layer;
      setActive("[data-layer]", state.layer, "layer");
      updateMaterials();
    });
  });
}

function updateProductAvailability(model) {
  const disparityButton = document.querySelector('[data-product="disparity"]');
  const hasDisparity = Boolean(model.disparity);
  disparityButton.disabled = !hasDisparity;
  if (!hasDisparity && state.product === "disparity") {
    state.product = "dsm";
    setActive("[data-product]", state.product, "product");
  }
}

function loadModel(modelKey) {
  const model = models[modelKey] || models.ours;
  updateProductAvailability(model);
  const product = products[state.product] || products.disparity;
  const meshPath = model[state.product] || model.dsm;
  container.dataset.status = `Loading ${model.label} ${product}`;

  loader.load(
    meshPath,
    (gltf) => {
      if (activeRoot) {
        scene.remove(activeRoot);
        disposeRoot(activeRoot);
      }

      activeRoot = gltf.scene;
      prepareRoot(activeRoot);
      frameRoot(activeRoot, !hasFramedOnce);
      hasFramedOnce = true;
      scene.add(activeRoot);
      container.dataset.status = "";
    },
    undefined,
    (error) => {
      container.innerHTML = "<p class=\"viewer-error\">The mesh asset could not be loaded.</p>";
      throw error;
    },
  );
}

function prepareRoot(root) {
  root.traverse((child) => {
    if (!child.isMesh) return;
    child.frustumCulled = false;
    applyIntenseColors(child.geometry);
    child.material = makeMaterial(child.geometry);
  });
}

function frameRoot(root, resetView) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());

  // X/Z are intentionally left untouched: every mesh is exported with the
  // same deterministic grid-center offset, so leaving them as-is keeps all
  // models (including LiDAR GT) spatially aligned with each other.
  root.position.y -= box.min.y;

  orbit.panLimit = Math.max(60, Math.max(size.x, size.z) * 0.55);

  if (resetView) {
    orbit.target.set(0, Math.max(8, size.y * 0.38), 0);
    orbit.radius = Math.max(190, Math.max(size.x, size.z) * 1.04);
  }

  orbit.target.x = THREE.MathUtils.clamp(orbit.target.x, -orbit.panLimit, orbit.panLimit);
  orbit.target.z = THREE.MathUtils.clamp(orbit.target.z, -orbit.panLimit, orbit.panLimit);
  updateCamera();
}

function makeMaterial(geometry) {
  const hasVertexColors = Boolean(geometry.getAttribute("color"));
  if (state.layer === "wire") {
    return new THREE.MeshBasicMaterial({
      color: 0x9fffd3,
      wireframe: true,
      side: THREE.DoubleSide,
    });
  }

  return new THREE.MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: hasVertexColors,
    side: THREE.DoubleSide,
  });
}

function applyIntenseColors(geometry) {
  const position = geometry.getAttribute("position");
  if (!position) return;

  const values = [];
  for (let index = 0; index < position.count; index += 1) {
    const value = position.getY(index);
    if (Number.isFinite(value)) {
      values.push(value);
    }
  }
  if (!values.length) return;

  values.sort((a, b) => a - b);
  const low = percentile(values, 0.01);
  const high = percentile(values, 0.99);
  const denom = Math.max(high - low, 1e-6);
  const colors = new Float32Array(position.count * 3);

  for (let index = 0; index < position.count; index += 1) {
    const raw = position.getY(index);
    const normalized = smootherstep(THREE.MathUtils.clamp((raw - low) / denom, 0, 1));
    const color = intenseRamp(normalized);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  }

  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

const hslScratch = { h: 0, s: 0, l: 0 };

function intenseRamp(value) {
  const stops = [
    0x12346b,
    0x0c83c5,
    0x22d6a2,
    0xf0d75b,
    0xf27b39,
    0xc8323f,
  ];
  const color = ramp(value, stops);
  color.getHSL(hslScratch);
  const s = THREE.MathUtils.clamp(hslScratch.s * 1.4 + 0.1, 0, 1);
  const l = THREE.MathUtils.clamp(0.5 + (hslScratch.l - 0.5) * 1.15, 0.1, 0.85);
  color.setHSL(hslScratch.h, s, l);
  return color;
}

function ramp(value, stops) {
  const scaled = THREE.MathUtils.clamp(value, 0, 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const local = scaled - index;
  const a = new THREE.Color(stops[index]);
  const b = new THREE.Color(stops[index + 1]);
  return a.lerp(b, local);
}

function smootherstep(value) {
  return value * value * value * (value * (value * 6 - 15) + 10);
}

function percentile(sortedValues, q) {
  if (sortedValues.length === 1) return sortedValues[0];
  const index = THREE.MathUtils.clamp(q, 0, 1) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sortedValues[lower];
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function updateMaterials() {
  if (!activeRoot) return;
  activeRoot.traverse((child) => {
    if (!child.isMesh) return;
    const previousMaterial = child.material;
    child.material = makeMaterial(child.geometry);
    if (previousMaterial) previousMaterial.dispose();
  });
}

function disposeRoot(root) {
  root.traverse((child) => {
    if (!child.isMesh) return;
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  });
}

function setActive(selector, value, key) {
  document.querySelectorAll(selector).forEach((button) => {
    const isActive = button.dataset[key] === value;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function installPointerControls() {
  const canvas = renderer.domElement;

  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  canvas.addEventListener("pointerdown", (event) => {
    const isPanRequest = event.button === 2 || event.shiftKey;
    orbit.panning = isPanRequest;
    orbit.dragging = !isPanRequest;
    orbit.lastX = event.clientX;
    orbit.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!orbit.dragging && !orbit.panning) return;
    const deltaX = event.clientX - orbit.lastX;
    const deltaY = event.clientY - orbit.lastY;
    orbit.lastX = event.clientX;
    orbit.lastY = event.clientY;

    if (orbit.panning) {
      const panScale = orbit.radius * 0.0016;
      const right = Math.cos(orbit.theta);
      const rightZ = -Math.sin(orbit.theta);
      const forwardX = Math.sin(orbit.theta);
      const forward = Math.cos(orbit.theta);
      orbit.target.x -= (right * deltaX - forwardX * deltaY) * panScale;
      orbit.target.z -= (rightZ * deltaX - forward * deltaY) * panScale;
      orbit.target.x = THREE.MathUtils.clamp(orbit.target.x, -orbit.panLimit, orbit.panLimit);
      orbit.target.z = THREE.MathUtils.clamp(orbit.target.z, -orbit.panLimit, orbit.panLimit);
    } else {
      orbit.theta -= deltaX * 0.006;
      orbit.phi = THREE.MathUtils.clamp(orbit.phi + deltaY * 0.006, orbit.minPhi, orbit.maxPhi);
    }
    updateCamera();
  });

  canvas.addEventListener("pointerup", (event) => {
    orbit.dragging = false;
    orbit.panning = false;
    canvas.releasePointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointercancel", () => {
    orbit.dragging = false;
    orbit.panning = false;
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    orbit.radius = THREE.MathUtils.clamp(orbit.radius + event.deltaY * 0.1, orbit.minRadius, orbit.maxRadius);
    updateCamera();
  }, { passive: false });
}

function updateCamera() {
  const sinPhi = Math.sin(orbit.phi);
  camera.position.set(
    orbit.target.x + orbit.radius * sinPhi * Math.sin(orbit.theta),
    orbit.target.y + orbit.radius * Math.cos(orbit.phi),
    orbit.target.z + orbit.radius * sinPhi * Math.cos(orbit.theta),
  );
  camera.lookAt(orbit.target);
}

function resize() {
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 560;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}
