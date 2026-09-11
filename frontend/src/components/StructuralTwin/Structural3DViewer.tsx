import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Eye, RotateCcw, ShieldAlert, Activity } from "lucide-react";

interface Structural3DViewerProps {
  stories: number;
  storyDeflectionsMm: number[]; // lateral displacement per story in mm
  yieldDisplacementMm: number;  // yield limit in mm
  currentTimeS: number;
  peakDisplacementMm: number;
  isYielded: boolean;
  buildingName: string;
}

export const Structural3DViewer: React.FC<Structural3DViewerProps> = ({
  stories,
  storyDeflectionsMm,
  yieldDisplacementMm,
  currentTimeS,
  isYielded,
  buildingName,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // Mesh refs for dynamic deformation
  const slabsRef = useRef<THREE.Group[]>([]);
  const columnsRef = useRef<{
    mesh: THREE.Mesh;
    baseJoint: THREE.Mesh;
    topJoint: THREE.Mesh;
    storyIdx: number;
    cornerIdx: number;
  }[]>([]);
  const groundGridRef = useRef<THREE.GridHelper | null>(null);

  const [cameraPreset, setCameraPreset] = useState<"iso" | "front" | "top">("iso");

  // Visual amplification factor so real structural drifts (e.g. 10-60 mm) are clearly visible
  const VISUAL_SCALE = 0.080; // 10mm -> 0.80 3D units for vivid visible deformation
  const FLOOR_WIDTH = 7.0;
  const FLOOR_DEPTH = 7.0;
  const STORY_H = 3.2;

  // Initialize Three.js Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    // 1. Scene: Light Architectural Off-White Canvas
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    scene.fog = new THREE.FogExp2(0xf8fafc, 0.012);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 200);
    const totalHeight = stories * STORY_H;
    camera.position.set(16, totalHeight * 0.75 + 4, 22);
    camera.lookAt(0, totalHeight * 0.45, 0);
    cameraRef.current = camera;

    // 3. Renderer with antialiasing
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // prevent going below ground
    controls.minDistance = 8;
    controls.maxDistance = 70;
    controls.target.set(0, totalHeight * 0.45, 0);
    controlsRef.current = controls;

    // 5. Lighting: Natural Studio White Light
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.9);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 2.0);
    mainLight.position.set(20, 35, 20);
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xe2e8f0, 1.0);
    fillLight.position.set(-20, 20, -20);
    scene.add(fillLight);

    // 6. Seismic Ground Grid & Foundation Pad (Light Engineering Grid)
    const grid = new THREE.GridHelper(30, 30, 0xcbd5e1, 0xe2e8f0);
    grid.position.y = -0.05;
    scene.add(grid);
    groundGridRef.current = grid;

    // Foundation Plate
    const foundGeo = new THREE.BoxGeometry(FLOOR_WIDTH + 2.0, 0.4, FLOOR_DEPTH + 2.0);
    const foundMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      roughness: 0.8,
      metalness: 0.1,
    });
    const foundMesh = new THREE.Mesh(foundGeo, foundMat);
    foundMesh.position.y = -0.2;
    scene.add(foundMesh);

    // Foundation edge outline
    const foundEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(foundGeo),
      new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.7 })
    );
    foundMesh.add(foundEdges);

    // 7. Animation Loop
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animFrameIdRef.current = requestAnimationFrame(animate);
    };
    animate();

    // 8. Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    const ro = new ResizeObserver(handleResize);
    ro.observe(container);

    return () => {
      ro.disconnect();
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [stories]);

  // Build / Rebuild Building Slabs & Columns Geometry when story count changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clean up old building meshes
    slabsRef.current.forEach((slab) => scene.remove(slab));
    slabsRef.current = [];

    columnsRef.current.forEach((col) => {
      scene.remove(col.mesh);
      scene.remove(col.baseJoint);
      scene.remove(col.topJoint);
    });
    columnsRef.current = [];

    // Shared Materials: Clean Architectural Chalk Slabs & Precision Slate Columns
    const slabGeo = new THREE.BoxGeometry(FLOOR_WIDTH, 0.22, FLOOR_DEPTH);
    const slabMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.1,
      roughness: 0.6,
      transparent: false,
    });
    const slabEdgeMat = new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.8 });
    const slabEdgesGeo = new THREE.EdgesGeometry(slabGeo);

    const jointGeo = new THREE.SphereGeometry(0.20, 16, 16);
    const columnGeo = new THREE.CylinderGeometry(0.12, 0.12, STORY_H, 12);

    for (let story = 0; story < stories; story++) {
      const floorY = (story + 1) * STORY_H;

      // Create Floor Slab Group
      const slabGroup = new THREE.Group();
      slabGroup.position.set(0, floorY, 0);

      const slabMesh = new THREE.Mesh(slabGeo, slabMat);
      const slabEdges = new THREE.LineSegments(slabEdgesGeo, slabEdgeMat);
      slabMesh.add(slabEdges);
      slabGroup.add(slabMesh);

      // Floor Level Marker
      scene.add(slabGroup);
      slabsRef.current.push(slabGroup);

      // Create 4 Columns per Story: Graphite Slate
      for (let c = 0; c < 4; c++) {
        const colMat = new THREE.MeshStandardMaterial({
          color: 0x334155, // Graphite Slate
          roughness: 0.35,
          metalness: 0.4,
        });
        const colMesh = new THREE.Mesh(columnGeo, colMat);

        const jointMat = new THREE.MeshStandardMaterial({
          color: 0x475569,
          roughness: 0.3,
          metalness: 0.5,
        });
        const baseJoint = new THREE.Mesh(jointGeo, jointMat);
        const topJoint = new THREE.Mesh(jointGeo, jointMat.clone());

        scene.add(colMesh);
        scene.add(baseJoint);
        scene.add(topJoint);

        columnsRef.current.push({
          mesh: colMesh,
          baseJoint,
          topJoint,
          storyIdx: story,
          cornerIdx: c,
        });
      }
    }
  }, [stories]);

  // Real-Time Dynamic Deformation Update
  useEffect(() => {
    if (!sceneRef.current) return;

    const halfW = (FLOOR_WIDTH - 0.8) / 2;
    const halfD = (FLOOR_DEPTH - 0.8) / 2;
    const cornerOffsets = [
      [-halfW, -halfD],
      [halfW, -halfD],
      [-halfW, halfD],
      [halfW, halfD],
    ];

    // Compute and apply lateral story drift to floor slabs
    slabsRef.current.forEach((slab, sIdx) => {
      const deflMm = storyDeflectionsMm[sIdx] || 0;
      const deflX = deflMm * VISUAL_SCALE;
      const targetY = (sIdx + 1) * STORY_H;
      slab.position.set(deflX, targetY, 0);
    });

    // Re-orient and shear columns between consecutive floor levels
    columnsRef.current.forEach((col) => {
      const { storyIdx, cornerIdx } = col;
      const [cx, cz] = cornerOffsets[cornerIdx];

      const baseDeflMm = storyIdx === 0 ? 0 : storyDeflectionsMm[storyIdx - 1] || 0;
      const topDeflMm = storyDeflectionsMm[storyIdx] || 0;

      const baseX = cx + baseDeflMm * VISUAL_SCALE;
      const baseY = storyIdx * STORY_H;
      const baseZ = cz;

      const topX = cx + topDeflMm * VISUAL_SCALE;
      const topY = (storyIdx + 1) * STORY_H;
      const topZ = cz;

      col.baseJoint.position.set(baseX, baseY, baseZ);
      col.topJoint.position.set(topX, topY, topZ);

      // Midpoint position
      col.mesh.position.set((baseX + topX) / 2, (baseY + topY) / 2, (baseZ + topZ) / 2);

      // Orientation vector
      const v = new THREE.Vector3(topX - baseX, topY - baseY, topZ - baseZ);
      const len = v.length();
      const up = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, v.clone().normalize());
      col.mesh.setRotationFromQuaternion(quat);
      col.mesh.scale.set(1, len / STORY_H, 1);

      // Color coding plastic yielding on top/base joint hinges
      const storyDriftMm = Math.abs(topDeflMm - baseDeflMm);
      const yieldPerStoryMm = yieldDisplacementMm / stories;
      const storyYielded = storyDriftMm >= yieldPerStoryMm;

      const mat = col.mesh.material as THREE.MeshStandardMaterial;
      const topMat = col.topJoint.material as THREE.MeshStandardMaterial;

      if (storyYielded) {
        // Red plastic hinge marker
        mat.color.setHex(0xdc2626);
        topMat.color.setHex(0xdc2626);
      } else {
        mat.color.setHex(0x334155);
        topMat.color.setHex(0x475569);
      }
    });
  }, [storyDeflectionsMm, yieldDisplacementMm, stories]);

  const setViewPreset = useCallback((preset: "iso" | "front" | "top") => {
    setCameraPreset(preset);
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    const totalH = stories * STORY_H;
    const centerY = totalH * 0.45;
    controls.target.set(0, centerY, 0);

    if (preset === "iso") {
      camera.position.set(16, totalH * 0.75 + 4, 22);
    } else if (preset === "front") {
      camera.position.set(0, centerY, 28);
    } else if (preset === "top") {
      camera.position.set(0.01, totalH + 20, 0);
    }
    controls.update();
  }, [stories]);

  const topFloorDisp = storyDeflectionsMm.length > 0 ? storyDeflectionsMm[storyDeflectionsMm.length - 1] : 0;

  return (
    <div className="relative w-full h-[460px] rounded border border-[#E2E8F0] bg-[#F8FAFC] shadow-xs overflow-hidden">
      {/* 3D WebGL Canvas Mount */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Header HUD: Clean Technical Overlay */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
        <div className="flex items-center space-x-2 bg-[#FFFFFF]/95 border border-[#CBD5E1] px-3 py-1.5 rounded shadow-xs pointer-events-auto">
          <div className="w-2 h-2 rounded-none bg-[#047857]" />
          <div>
            <div className="text-xs font-mono font-bold text-[#0F172A] tracking-tight flex items-center gap-1.5">
              <span>{buildingName}</span>
              <span className="text-[10px] px-1 py-0.2 rounded bg-[#F1F5F9] text-[#334155] border border-[#CBD5E1]">
                {stories} STORIES
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#64748B]">
              t = <span className="text-[#0F172A] font-semibold">{currentTimeS.toFixed(2)} s</span> · ROOF DRIFT:{" "}
              <span className={Math.abs(topFloorDisp) > yieldDisplacementMm ? "text-[#DC2626] font-bold" : "text-[#047857] font-bold"}>
                {topFloorDisp.toFixed(1)} mm
              </span>
            </div>
          </div>
        </div>

        {/* State Badge: Clean Technical Indicator */}
        <div className="flex items-center space-x-2 pointer-events-auto">
          <div
            className={`px-2.5 py-1 rounded border text-xs font-mono font-semibold flex items-center space-x-1.5 ${
              isYielded
                ? "bg-[#FEF2F2] border-[#FCA5A5] text-[#DC2626]"
                : "bg-[#ECFDF5] border-[#A7F3D0] text-[#047857]"
            }`}
          >
            {isYielded ? <ShieldAlert size={13} /> : <Activity size={13} />}
            <span>{isYielded ? "PLASTIC YIELD (HINGE FORMED)" : "ELASTIC RESPONSE"}</span>
          </div>
        </div>
      </div>

      {/* Floating Bottom View Controls HUD */}
      <div className="absolute bottom-3 left-3 flex items-center space-x-1.5 pointer-events-auto bg-[#FFFFFF]/95 p-1 rounded border border-[#CBD5E1] shadow-xs">
        <span className="text-[10px] font-mono text-[#64748B] px-1.5 flex items-center gap-1">
          <Eye size={11} className="text-[#334155]" /> VIEW:
        </span>
        <button
          onClick={() => setViewPreset("iso")}
          className={`px-2 py-0.5 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "iso" ? "bg-[#0F172A] text-white font-semibold" : "text-[#475569] hover:text-[#0F172A]"
          }`}
        >
          3D Orbit
        </button>
        <button
          onClick={() => setViewPreset("front")}
          className={`px-2 py-0.5 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "front" ? "bg-[#0F172A] text-white font-semibold" : "text-[#475569] hover:text-[#0F172A]"
          }`}
        >
          Elevation
        </button>
        <button
          onClick={() => setViewPreset("top")}
          className={`px-2 py-0.5 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "top" ? "bg-[#0F172A] text-white font-semibold" : "text-[#475569] hover:text-[#0F172A]"
          }`}
        >
          Plan
        </button>
        <button
          onClick={() => setViewPreset("iso")}
          title="Reset Camera"
          className="p-1 text-[#64748B] hover:text-[#0F172A] cursor-pointer transition ml-0.5"
        >
          <RotateCcw size={12} />
        </button>
      </div>
    </div>
  );
};

export default Structural3DViewer;
