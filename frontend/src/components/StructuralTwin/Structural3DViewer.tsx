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
  peakDisplacementMm,
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
  // on a building of height ~15-20 units without looking unrealistically distorted
  const VISUAL_SCALE = 0.045; // 10mm -> ~0.45 3D units
  const FLOOR_WIDTH = 7.0;
  const FLOOR_DEPTH = 7.0;
  const STORY_H = 3.2;

  // Initialize Three.js Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c16);
    scene.fog = new THREE.FogExp2(0x080c16, 0.015);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 200);
    const totalHeight = stories * STORY_H;
    camera.position.set(16, totalHeight * 0.75 + 4, 22);
    camera.lookAt(0, totalHeight * 0.45, 0);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
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

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0x1e293b, 1.8);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 2.2);
    mainLight.position.set(20, 30, 20);
    scene.add(mainLight);

    // Subtle neon rim lights for SciML aesthetic
    const cyanRim = new THREE.DirectionalLight(0x00f0ff, 2.0);
    cyanRim.position.set(-20, 15, -15);
    scene.add(cyanRim);

    const purpleAccent = new THREE.PointLight(0x8a2be2, 2.5, 40);
    purpleAccent.position.set(15, totalHeight + 5, -10);
    scene.add(purpleAccent);

    // 6. Seismic Ground Grid & Foundation Pad
    const grid = new THREE.GridHelper(30, 30, 0x00f0ff, 0x1e293b);
    grid.position.y = -0.05;
    scene.add(grid);
    groundGridRef.current = grid;

    // Foundation Plate
    const foundGeo = new THREE.BoxGeometry(FLOOR_WIDTH + 2.0, 0.4, FLOOR_DEPTH + 2.0);
    const foundMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.4,
      metalness: 0.6,
    });
    const foundMesh = new THREE.Mesh(foundGeo, foundMat);
    foundMesh.position.y = -0.2;
    scene.add(foundMesh);

    // Foundation edge outline
    const foundEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(foundGeo),
      new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.4 })
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

    const halfW = (FLOOR_WIDTH - 0.8) / 2;
    const halfD = (FLOOR_DEPTH - 0.8) / 2;
    const cornerOffsets = [
      [-halfW, -halfD], // 0: Back-Left
      [halfW, -halfD],  // 1: Back-Right
      [-halfW, halfD],  // 2: Front-Left
      [halfW, halfD],   // 3: Front-Right
    ];

    // Shared Materials
    const slabGeo = new THREE.BoxGeometry(FLOOR_WIDTH, 0.22, FLOOR_DEPTH);
    const slabMat = new THREE.MeshStandardMaterial({
      color: 0x111c33,
      metalness: 0.5,
      roughness: 0.25,
      transparent: true,
      opacity: 0.85,
    });
    const slabEdgeMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.75 });
    const slabEdgesGeo = new THREE.EdgesGeometry(slabGeo);

    const jointGeo = new THREE.SphereGeometry(0.22, 16, 16);
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

      // Create 4 Columns per Story
      for (let c = 0; c < 4; c++) {
        const [cx, cz] = cornerOffsets[c];

        // Column Mesh (initial upright cylinder)
        const colMat = new THREE.MeshStandardMaterial({
          color: 0x00f0ff,
          metalness: 0.6,
          roughness: 0.2,
          emissive: 0x003344,
          emissiveIntensity: 0.2,
        });
        const colMesh = new THREE.Mesh(columnGeo, colMat);

        // Spherical Joint Nodes
        const jointMat = new THREE.MeshStandardMaterial({
          color: 0x00f0ff,
          metalness: 0.7,
          roughness: 0.2,
          emissive: 0x003344,
          emissiveIntensity: 0.4,
        });
        const baseJoint = new THREE.Mesh(jointGeo, jointMat);
        const topJoint = new THREE.Mesh(jointGeo, jointMat);

        baseJoint.position.set(cx, story * STORY_H, cz);
        topJoint.position.set(cx, floorY, cz);

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

    // Adjust camera target to center of new building
    if (controlsRef.current && cameraRef.current) {
      const totalH = stories * STORY_H;
      controlsRef.current.target.set(0, totalH * 0.45, 0);
      controlsRef.current.update();
    }
  }, [stories]);

  // Update Mesh Positions & Column Orientations whenever lateral deflections change
  useEffect(() => {
    if (slabsRef.current.length === 0 || columnsRef.current.length === 0) return;

    const halfW = (FLOOR_WIDTH - 0.8) / 2;
    const halfD = (FLOOR_DEPTH - 0.8) / 2;
    const cornerOffsets = [
      [-halfW, -halfD],
      [halfW, -halfD],
      [-halfW, halfD],
      [halfW, halfD],
    ];

    // Colors
    const elasticColor = new THREE.Color(0x00f0ff);
    const plasticColor = new THREE.Color(0xff2a6d);
    const elasticEmissive = new THREE.Color(0x003344);
    const plasticEmissive = new THREE.Color(0xaa0033);

    // 1. Update Slab Translations
    slabsRef.current.forEach((slab, idx) => {
      const dispMm = storyDeflectionsMm[idx] || 0;
      const xShift = dispMm * VISUAL_SCALE;
      slab.position.x = xShift;
    });

    // 2. Update Columns connecting Level (story-1) to Level (story)
    columnsRef.current.forEach((col) => {
      const s = col.storyIdx;
      const prevDispMm = s === 0 ? 0 : storyDeflectionsMm[s - 1] || 0;
      const currDispMm = storyDeflectionsMm[s] || 0;

      const prevX = prevDispMm * VISUAL_SCALE;
      const currX = currDispMm * VISUAL_SCALE;

      const [cx, cz] = cornerOffsets[col.cornerIdx];

      const p1 = new THREE.Vector3(cx + prevX, s * STORY_H, cz);
      const p2 = new THREE.Vector3(cx + currX, (s + 1) * STORY_H, cz);

      // Position joints
      col.baseJoint.position.copy(p1);
      col.topJoint.position.copy(p2);

      // Midpoint & orientation for cylinder
      const midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
      col.mesh.position.copy(midPoint);

      const dir = new THREE.Vector3().subVectors(p2, p1);
      const len = dir.length();
      dir.normalize();

      // Orient cylinder (cylinder default points along +Y)
      const up = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
      col.mesh.quaternion.copy(quat);
      col.mesh.scale.set(1, len / STORY_H, 1);

      // Plasticity check: Story drift
      const interstoryDriftMm = Math.abs(currDispMm - prevDispMm);
      const yielded = interstoryDriftMm > yieldDisplacementMm;

      // Color transition
      const colMat = col.mesh.material as THREE.MeshStandardMaterial;
      const jMat = col.baseJoint.material as THREE.MeshStandardMaterial;
      const targetCol = yielded ? plasticColor : elasticColor;
      const targetEmis = yielded ? plasticEmissive : elasticEmissive;
      const targetIntensity = yielded ? 0.85 : 0.25;

      colMat.color.lerp(targetCol, 0.25);
      colMat.emissive.lerp(targetEmis, 0.25);
      colMat.emissiveIntensity = targetIntensity;

      jMat.color.lerp(targetCol, 0.25);
      jMat.emissive.lerp(targetEmis, 0.25);
      jMat.emissiveIntensity = targetIntensity * 1.2;
    });
  }, [storyDeflectionsMm, yieldDisplacementMm]);

  // Camera Presets
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
  const maxStoryDrift = Math.max(
    ...storyDeflectionsMm.map((d, i) => (i === 0 ? Math.abs(d) : Math.abs(d - storyDeflectionsMm[i - 1])))
  );

  return (
    <div className="relative w-full h-[460px] rounded-xl overflow-hidden border border-white/10 bg-[#080C16] shadow-2xl">
      {/* 3D WebGL Canvas Mount */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Header HUD */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
        <div className="flex items-center space-x-3 bg-[#0E1526]/85 backdrop-blur-md px-3.5 py-2 rounded-lg border border-white/10 pointer-events-auto shadow-lg">
          <div className="w-2.5 h-2.5 rounded-full bg-[#00F0FF] animate-pulse" />
          <div>
            <div className="text-xs font-mono font-bold text-white tracking-wider flex items-center gap-1.5">
              <span>{buildingName.toUpperCase()}</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-cyan-400">
                {stories} STORIES
              </span>
            </div>
            <div className="text-[10px] font-mono text-slate-400">
              TIME: <span className="text-white font-bold">{currentTimeS.toFixed(2)}s</span> | ROOF DRIFT:{" "}
              <span className={Math.abs(topFloorDisp) > yieldDisplacementMm ? "text-[#FF2A6D] font-bold" : "text-cyan-400 font-bold"}>
                {topFloorDisp.toFixed(1)} mm
              </span>
            </div>
          </div>
        </div>

        {/* State Badge */}
        <div className="flex items-center space-x-2 pointer-events-auto">
          <div
            className={`px-3 py-1.5 rounded-lg border text-xs font-mono font-bold flex items-center space-x-1.5 backdrop-blur-md transition-all duration-300 ${
              isYielded
                ? "bg-[#FF2A6D]/20 border-[#FF2A6D] text-[#FF2A6D] shadow-[0_0_15px_rgba(255,42,109,0.35)]"
                : "bg-[#00F0FF]/15 border-[#00F0FF]/60 text-[#00F0FF] shadow-[0_0_12px_rgba(0,240,255,0.25)]"
            }`}
          >
            {isYielded ? <ShieldAlert size={14} /> : <Activity size={14} />}
            <span>{isYielded ? "PLASTIC YIELD (HINGE FORMED)" : "ELASTIC VIBRATION"}</span>
          </div>
        </div>
      </div>

      {/* Floating Bottom View Controls HUD */}
      <div className="absolute bottom-4 left-4 flex items-center space-x-2 pointer-events-auto bg-[#0E1526]/85 backdrop-blur-md p-1.5 rounded-lg border border-white/10 shadow-lg">
        <span className="text-[10px] font-mono text-slate-400 px-2 flex items-center gap-1">
          <Eye size={12} className="text-cyan-400" /> VIEW:
        </span>
        <button
          onClick={() => setViewPreset("iso")}
          className={`px-2.5 py-1 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "iso" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-white"
          }`}
        >
          3D Orbit
        </button>
        <button
          onClick={() => setViewPreset("front")}
          className={`px-2.5 py-1 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "front" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-white"
          }`}
        >
          Elevation
        </button>
        <button
          onClick={() => setViewPreset("top")}
          className={`px-2.5 py-1 text-xs font-mono rounded cursor-pointer transition ${
            cameraPreset === "top" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-white"
          }`}
        >
          Plan
        </button>
        <button
          onClick={() => setViewPreset("iso")}
          title="Reset Orbit Camera"
          className="p-1 text-slate-400 hover:text-cyan-300 cursor-pointer transition ml-1"
        >
          <RotateCcw size={13} />
        </button>
      </div>

      {/* Floating Story Drift Legend HUD */}
      <div className="absolute bottom-4 right-4 bg-[#0E1526]/85 backdrop-blur-md px-3.5 py-2 rounded-lg border border-white/10 text-[11px] font-mono pointer-events-auto space-y-1 shadow-lg">
        <div className="flex items-center justify-between gap-3">
          <span className="text-slate-400">Peak Response:</span>
          <span className="text-white font-bold">{peakDisplacementMm.toFixed(1)} mm</span>
        </div>
        <div className="flex items-center justify-between gap-3">
          <span className="text-slate-400">Max Story Drift:</span>
          <span className={maxStoryDrift > yieldDisplacementMm ? "text-[#FF2A6D] font-bold" : "text-cyan-400 font-bold"}>
            {maxStoryDrift.toFixed(1)} mm
          </span>
        </div>
        <div className="flex items-center gap-3 pt-1 border-t border-white/10 text-[10px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#00F0FF] shadow-[0_0_6px_#00F0FF]" />
            <span className="text-slate-300">Elastic</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#FF2A6D] shadow-[0_0_6px_#FF2A6D]" />
            <span className="text-slate-300">Yielded Hinge</span>
          </div>
        </div>
      </div>
    </div>
  );
};
