import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const LIGHT_CLASS_COLORS = [0x16a765, 0x2583f5, 0xef4444, 0x8b5cf6, 0xf59e0b];
const DARK_CLASS_COLORS = [0x39d98a, 0x55a7ff, 0xff6875, 0xb49aff, 0xffc15a];

function textSprite(text, color = "#172554", dark = false) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 512;
  canvas.height = 128;
  context.font = "600 30px Inter, Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = dark ? "rgba(25,35,55,0.97)" : "rgba(255,255,255,0.94)";
  context.roundRect(5, 8, 502, 112, 22);
  context.fill();
  context.strokeStyle = dark ? "rgba(150,170,205,0.48)" : "rgba(99,102,241,0.35)";
  context.lineWidth = 3;
  context.stroke();
  context.fillStyle = dark ? "#f0f5ff" : color;
  const labelText = String(text).slice(0, 40);
  const lines = labelText.length > 22
    ? [labelText.slice(0, 22), labelText.slice(22)]
    : [labelText];
  lines.forEach((line, index) => context.fillText(line, 256, lines.length === 1 ? 64 : 43 + index * 42));

  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true }),
  );
  sprite.scale.set(2, 0.84, 1);
  return sprite;
}

function pointTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.beginPath();
  context.arc(32, 32, 27, 0, Math.PI * 2);
  context.fillStyle = "#ffffff";
  context.fill();
  context.lineWidth = 5;
  context.strokeStyle = "#ffffff";
  context.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function nodePositions(nodes) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const positions = new Map();
  let leaf = 0;

  function place(id) {
    const node = byId.get(id);
    if (!node) return null;
    const left = node.left !== -1 ? place(node.left) : null;
    const right = node.right !== -1 ? place(node.right) : null;
    const x = left && right ? (left.x + right.x) / 2 : left?.x ?? right?.x ?? leaf++ * 2.2;
    const point = new THREE.Vector3(
      x,
      (Math.max(0, ...nodes.map((item) => item.depth)) - node.depth) * 4,
      node.depth * 0.75,
    );
    positions.set(id, point);
    return { x };
  }

  if (byId.has(0)) place(0);
  return positions;
}

export default function Visualization3D({
  mode,
  dark = false,
  nodes = [],
  rows = [],
  features = [],
  allFeatures = [],
  target,
  classNames = [],
  selectedNode,
  onSelectNode,
}) {
  const hostRef = useRef(null);
  const resetViewRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    const scene = new THREE.Scene();
    const sceneBackground = dark ? "#111827" : "#f8faff";
    scene.background = new THREE.Color(sceneBackground);
    const camera = new THREE.PerspectiveCamera(43, 1, 0.1, 500);
    camera.position.set(13, 12, 19);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(sceneBackground);
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 7;
    controls.maxDistance = 70;
    controls.zoomSpeed = 1.15;
    controls.target.set(0, 3, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa8c0, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 1.8);
    light.position.set(8, 15, 10);
    scene.add(light);

    const grid = new THREE.GridHelper(14, 14, dark ? 0x53627b : 0xb5c2d7, dark ? 0x2b374b : 0xe0e7f1);
    grid.position.y = -0.03;
    scene.add(grid);

    const clickable = [];
    const disposables = [];
    const classColor = (name) => {
      const index = Math.max(0, classNames.indexOf(name));
      const colors = dark ? DARK_CLASS_COLORS : LIGHT_CLASS_COLORS;
      return colors[index % colors.length];
    };

    if (mode === "tree") {
      const positions = nodePositions(nodes);
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));
      const centerX =
        (Math.min(...[...positions.values()].map((point) => point.x), 0) +
          Math.max(...[...positions.values()].map((point) => point.x), 0)) /
        2;

      for (const point of positions.values()) point.x -= centerX;

      for (const node of nodes) {
        const point = positions.get(node.id);
        if (!point) continue;

        for (const childId of [node.left, node.right]) {
          const child = nodeMap.get(childId);
          const childPoint = positions.get(childId);
          if (!child || !childPoint) continue;
          const start = point.clone();
          const end = childPoint.clone();
          const edgeGeometry = new THREE.BufferGeometry().setFromPoints([start, end]);
          const edgeMaterial = new THREE.LineBasicMaterial({
            color: childId === node.left ? 0x22a06b : 0x4385f5,
            transparent: true,
            opacity: 0.8,
          });
          const edge = new THREE.Line(edgeGeometry, edgeMaterial);
          scene.add(edge);
          disposables.push(edgeGeometry, edgeMaterial);
        }
      }

      for (const node of nodes) {
        const point = positions.get(node.id);
        if (!point) continue;
        const color = classColor(node.prediction_name);
        const geometry = new THREE.SphereGeometry(node.id === selectedNode ? 0.48 : 0.36, 24, 18);
        const material = new THREE.MeshStandardMaterial({
          color: node.id === selectedNode ? 0xf59e0b : color,
          roughness: 0.3,
          metalness: 0.05,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(point);
        mesh.userData.nodeId = node.id;
        scene.add(mesh);
        clickable.push(mesh);
        disposables.push(geometry, material);

        const label = textSprite(
          node.is_leaf
            ? `${node.prediction_name} · ${node.samples}`
            : `${node.feature} ≤ ${Number(node.threshold).toPrecision(4)}`,
          "#172554",
          dark,
        );
        label.position.set(point.x, point.y + 0.85, point.z);
        scene.add(label);
        disposables.push(label.material.map, label.material);
      }

      const points = [...positions.values()];
      const treeWidth =
        Math.max(...points.map((point) => point.x), 0) -
        Math.min(...points.map((point) => point.x), 0);
      const treeHeight = Math.max(...points.map((point) => point.y), 1);
      grid.scale.x = Math.max(1, treeWidth / 14);
      const aspect = host.clientWidth / Math.max(host.clientHeight, 1);
      const halfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const horizontalFit = (treeWidth / 2 + 1.8) / (halfFov * Math.max(aspect, 0.35));
      const verticalFit = (treeHeight / 2 + 1.25) / halfFov;
      const distance = Math.max(horizontalFit, verticalFit, 14) * 1.08;
      controls.target.set(0, treeHeight / 2, 0);
      camera.position.set(0, treeHeight / 2 + distance * 0.18, distance);
    } else {
      controls.target.set(0, 0, 0);
      // Frame the full normalized feature-space cube with breathing room.
      // The overview canvas can grow to the viewport height, so keep the
      // initial camera far enough back to show the complete surface volume.
      camera.position.set(15.7, 12.8, 20);
      const [xFeature, yFeature, zFeature] = features.slice(0, 3);
      const numericRows = rows.filter(
        (row) =>
          [xFeature, yFeature, zFeature].every(
            (feature) => feature && Number.isFinite(Number(row[feature])),
          ),
      );
      const ranges = [xFeature, yFeature, zFeature].map((feature) => {
        const values = numericRows.map((row) => Number(row[feature]));
        if (!values.length) return [0, 1];
        const min = Math.min(...values);
        const max = Math.max(...values);
        const padding = Math.max((max - min) * 0.08, Math.abs(min || 1) * 0.04, 0.1);
        return [min - padding, max + padding];
      });
      const toAxis = (value, [min, max]) =>
        ((Number(value) - min) / (max - min || 1) - 0.5) * 12;
      const defaults = Object.fromEntries(
        allFeatures.map((feature) => {
          const values = rows.map((row) => row[feature]);
          const numeric = values.map(Number);
          if (numeric.length && numeric.every(Number.isFinite)) {
            numeric.sort((a, b) => a - b);
            return [feature, numeric[Math.floor(numeric.length / 2)]];
          }
          const categories = [...new Set(values.map(String))].sort();
          return [feature, Math.floor((categories.length - 1) / 2)];
        }),
      );
      const nodeMap = new Map(nodes.map((node) => [node.id, node]));

      if (mode === "surface") {
        const leafRegions = [];
        function collectRegions(nodeId, bounds) {
          const node = nodeMap.get(nodeId);
          if (!node) return;
          if (node.is_leaf) {
            leafRegions.push({ node, bounds });
            return;
          }
          const axisIndex = features.indexOf(node.feature);
          if (axisIndex === -1) {
            const hiddenValue = Number(defaults[node.feature] ?? 0);
            collectRegions(hiddenValue <= node.threshold ? node.left : node.right, bounds);
            return;
          }
          const leftBounds = bounds.map((range) => [...range]);
          const rightBounds = bounds.map((range) => [...range]);
          leftBounds[axisIndex][1] = Math.min(leftBounds[axisIndex][1], Number(node.threshold));
          rightBounds[axisIndex][0] = Math.max(rightBounds[axisIndex][0], Number(node.threshold));
          if (leftBounds[axisIndex][0] < leftBounds[axisIndex][1]) collectRegions(node.left, leftBounds);
          if (rightBounds[axisIndex][0] < rightBounds[axisIndex][1]) collectRegions(node.right, rightBounds);
        }
        collectRegions(0, ranges.map((range) => [...range]));

        for (const { node, bounds } of leafRegions) {
          const center = bounds.map(([min, max], axis) => toAxis((min + max) / 2, ranges[axis]));
          const dimensions = bounds.map(([min, max], axis) => Math.max(0.035, ((max - min) / (ranges[axis][1] - ranges[axis][0] || 1)) * 12 - 0.035));
          const geometry = new THREE.BoxGeometry(...dimensions);
          const material = new THREE.MeshBasicMaterial({
            color: classColor(node.prediction_name),
            transparent: true,
            opacity: 0.055,
            depthWrite: false,
            side: THREE.DoubleSide,
          });
          const region = new THREE.Mesh(geometry, material);
          region.position.set(...center);
          region.renderOrder = 1;
          scene.add(region);
          disposables.push(geometry, material);

          const edgeGeometry = new THREE.EdgesGeometry(geometry);
          const edgeMaterial = new THREE.LineBasicMaterial({
            color: classColor(node.prediction_name),
            transparent: true,
            opacity: 0.72,
          });
          const outline = new THREE.LineSegments(edgeGeometry, edgeMaterial);
          outline.position.copy(region.position);
          outline.renderOrder = 2;
          scene.add(outline);
          disposables.push(edgeGeometry, edgeMaterial);
        }
      }

      const rowsByClass = new Map();
      for (const row of numericRows) {
        const className = String(row[target]);
        if (!rowsByClass.has(className)) rowsByClass.set(className, []);
        rowsByClass.get(className).push(row);
      }
      for (const [className, classRows] of rowsByClass) {
        const positions = new Float32Array(classRows.length * 3);
        classRows.forEach((row, index) => {
          positions[index * 3] = toAxis(row[xFeature], ranges[0]);
          positions[index * 3 + 1] = toAxis(row[yFeature], ranges[1]);
          positions[index * 3 + 2] = toAxis(row[zFeature], ranges[2]);
        });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.computeBoundingSphere();
        const material = new THREE.PointsMaterial({
          color: classColor(className),
          size: 13,
          sizeAttenuation: false,
          transparent: false,
          toneMapped: false,
          map: pointTexture(),
          alphaTest: 0.35,
        });
        const dots = new THREE.Points(geometry, material);
        dots.renderOrder = 4;
        scene.add(dots);
        disposables.push(geometry, material, material.map);
      }

      const axisLabels = [
        [xFeature, new THREE.Vector3(6.8, -0.5, 0)],
        [yFeature, new THREE.Vector3(0, 6.8, 0)],
        [zFeature, new THREE.Vector3(0, -0.5, 6.8)],
      ];
      for (const [name, position] of axisLabels) {
        const label = textSprite(name || "Numeric feature", "#172554", dark);
        label.position.copy(position);
        label.scale.set(4, 1, 1);
        scene.add(label);
        disposables.push(label.material.map, label.material);
      }
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    function handleClick(event) {
      if (mode !== "tree" || !onSelectNode) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(clickable)[0];
      if (hit) onSelectNode(nodes.find((node) => node.id === hit.object.userData.nodeId));
    }
    renderer.domElement.addEventListener("dblclick", handleClick);

    let animationFrame = 0;
    function resize() {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(360, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    controls.update();
    controls.saveState();
    resetViewRef.current = () => controls.reset();

    function render() {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    }
    render();

    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      resetViewRef.current = null;
      renderer.domElement.removeEventListener("dblclick", handleClick);
      for (const item of disposables) item?.dispose?.();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [mode, dark, nodes, rows, features, allFeatures, target, classNames, selectedNode, onSelectNode]);

  return (
    <div className={`visualization-3d-frame mode-${mode} ${dark ? "theme-dark" : ""}`}>
      <div
        ref={hostRef}
        className="visualization-3d-canvas"
        role="img"
        aria-label={mode === "tree" ? "Interactive 3D decision tree. Double-click a node to inspect it." : mode === "surface" ? "Interactive 3D decision regions with colored samples." : "Interactive 3D scatter plot with class-colored sample points."}
      />
      <div className="visualization-toolbar">
        <span>Drag to rotate <i>·</i> scroll to zoom</span>
        <button type="button" onClick={() => resetViewRef.current?.()}>Reset view</button>
      </div>
    </div>
  );
}
