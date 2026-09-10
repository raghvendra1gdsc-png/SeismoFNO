import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RotateCcw, AlertCircle } from "lucide-react";
import type { LiveEarthquakeEvent, DemoSimulationResponse } from "../../api/researchDemoApi";

interface LiveSeismic3DViewportProps {
  selectedEvent: LiveEarthquakeEvent | null;
  visualScale: number;
  onChangeVisualScale: (scale: number) => void;
  isSimulating: boolean;
  simResult: DemoSimulationResponse | null;
}

export const LiveSeismic3DViewport: React.FC<LiveSeismic3DViewportProps> = ({
  selectedEvent,
  visualScale,
  onChangeVisualScale,
  isSimulating,
  simResult,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  const [hasWebGL, setHasWebGL] = useState<boolean>(true);
  const [cameraPreset, setCameraPreset] = useState<"iso" | "front" | "top">("iso");
  const [currentDriftMm, setCurrentDriftMm] = useState<number>(0.0);
  const [peakRoofDispMm, setPeakRoofDispMm] = useState<number>(0.0);

  // Model References
  const slabsRef = useRef<THREE.Group[]>([]);
  const columnMeshesRef = useRef<{
    mesh: THREE.Mesh;
    storyIdx: number;
    colX: number;
    colZ: number;
    height: number;
    baseY: number;
  }[]>([]);
  const nodeSpheresRef = useRef<THREE.Mesh[]>([]);
  const waveLinesRef = useRef<THREE.Line[]>([]);
  const epicenterGroupRef = useRef<THREE.Group | null>(null);
  const targetCamPosRef = useRef<THREE.Vector3 | null>(null);

  // Camera Presets (approx x=7, y=5, z=8 looking toward model)
  const CAM_PRESETS = {
    iso: new THREE.Vector3(7.2, 5.0, 8.2),
    front: new THREE.Vector3(0.0, 4.4, 12.0),
    top: new THREE.Vector3(0.0, 15.5, 0.001),
  };

  const STORY_COUNT = 3;
  const STORY_H = 2.8;
  const BAY_W = 5.4;
  const BAY_D = 5.4;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
      if (!gl) {
        setHasWebGL(false);
        return;
      }
    } catch {
      setHasWebGL(false);
      return;
    }

    const width = container.clientWidth || 700;
    const height = container.clientHeight || 480;

    // 1. Scene with deep forest / petrol graphite background & soft atmospheric fog
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1714);
    scene.fog = new THREE.FogExp2(0x07110f, 0.024);
    sceneRef.current = scene;

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 120);
    camera.position.copy(CAM_PRESETS.iso);
    cameraRef.current = camera;

    // 3. High-performance renderer with natural shadow/depth
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Orbit Controls with smooth damping
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.target.set(0, 3.8, 0);
    controls.maxPolarAngle = Math.PI / 2 + 0.01;
    controls.minDistance = 4.5;
    controls.maxDistance = 26.0;
    controlsRef.current = controls;

    // 5. Architectural / Studio Lighting (Warm Key, Petrol Fill, Soft Rim)
    const ambientLight = new THREE.AmbientLight(0x17483a, 1.1);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfffdf7, 1.3);
    keyLight.position.set(9, 16, 11);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 35;
    keyLight.shadow.camera.left = -8;
    keyLight.shadow.camera.right = 8;
    keyLight.shadow.camera.top = 8;
    keyLight.shadow.camera.bottom = -8;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x73e6b5, 0.28);
    fillLight.position.set(-9, 7, -9);
    scene.add(fillLight);

    // 6. Ground Plane & Subtle Engineering Grid
    const groundGeo = new THREE.PlaneGeometry(36, 36);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x07110f,
      roughness: 0.9,
      metalness: 0.1,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);

    const gridHelper = new THREE.GridHelper(30, 30, 0x17483a, 0x101d19);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    // 7. Foundation Footings
    const footingGeo = new THREE.BoxGeometry(0.85, 0.28, 0.85);
    const footingMat = new THREE.MeshStandardMaterial({
      color: 0x14241f,
      roughness: 0.7,
      metalness: 0.25,
    });
    const cornerOffsets = [
      [-BAY_W / 2, -BAY_D / 2],
      [BAY_W / 2, -BAY_D / 2],
      [-BAY_W / 2, BAY_D / 2],
      [BAY_W / 2, BAY_D / 2],
    ];

    cornerOffsets.forEach(([fx, fz]) => {
      const footing = new THREE.Mesh(footingGeo, footingMat);
      footing.position.set(fx, 0.14, fz);
      footing.receiveShadow = true;
      footing.castShadow = true;
      scene.add(footing);
    });

    // 8. Structural 3-Story Frame (Warm Graphite Metallic with Mint Accents)
    const slabs: THREE.Group[] = [];
    const columns: typeof columnMeshesRef.current = [];
    const nodeSpheres: THREE.Mesh[] = [];

    const slabMat = new THREE.MeshStandardMaterial({
      color: 0x121f1b,
      roughness: 0.5,
      metalness: 0.3,
      transparent: true,
      opacity: 0.9,
    });
    const slabEdgeMat = new THREE.LineBasicMaterial({ color: 0x73e6b5, transparent: true, opacity: 0.4 });
    const columnMat = new THREE.MeshStandardMaterial({
      color: 0x2b3833,
      roughness: 0.4,
      metalness: 0.35,
    });
    const beamMat = new THREE.MeshStandardMaterial({
      color: 0x222e29,
      roughness: 0.45,
      metalness: 0.3,
    });
    const nodeMat = new THREE.MeshStandardMaterial({
      color: 0x73e6b5,
      emissive: 0x73e6b5,
      emissiveIntensity: 0.3,
      roughness: 0.25,
    });

    const nodeGeo = new THREE.SphereGeometry(0.12, 16, 16);

    for (let story = 1; story <= STORY_COUNT; story++) {
      const floorY = story * STORY_H;
      const floorGroup = new THREE.Group();
      floorGroup.position.set(0, floorY, 0);

      // Floor Slab
      const slabMesh = new THREE.Mesh(new THREE.BoxGeometry(BAY_W + 0.4, 0.18, BAY_D + 0.4), slabMat);
      slabMesh.castShadow = true;
      slabMesh.receiveShadow = true;
      floorGroup.add(slabMesh);

      // Wireframe Outline
      const slabEdges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(BAY_W + 0.4, 0.18, BAY_D + 0.4)),
        slabEdgeMat
      );
      floorGroup.add(slabEdges);

      // Perimeter Beams
      const beamGeoX = new THREE.BoxGeometry(BAY_W, 0.22, 0.22);
      const beamGeoZ = new THREE.BoxGeometry(0.22, 0.22, BAY_D);

      const b1 = new THREE.Mesh(beamGeoX, beamMat);
      b1.position.set(0, -0.1, -BAY_D / 2);
      b1.castShadow = true;
      floorGroup.add(b1);

      const b2 = new THREE.Mesh(beamGeoX, beamMat);
      b2.position.set(0, -0.1, BAY_D / 2);
      b2.castShadow = true;
      floorGroup.add(b2);

      const b3 = new THREE.Mesh(beamGeoZ, beamMat);
      b3.position.set(-BAY_W / 2, -0.1, 0);
      b3.castShadow = true;
      floorGroup.add(b3);

      const b4 = new THREE.Mesh(beamGeoZ, beamMat);
      b4.position.set(BAY_W / 2, -0.1, 0);
      b4.castShadow = true;
      floorGroup.add(b4);

      // Node Spheres
      cornerOffsets.forEach(([cx, cz]) => {
        const node = new THREE.Mesh(nodeGeo, nodeMat);
        node.position.set(cx, 0, cz);
        floorGroup.add(node);
        nodeSpheres.push(node);
      });

      scene.add(floorGroup);
      slabs.push(floorGroup);

      // 4 Story Columns Below This Floor
      const colHeight = STORY_H;
      const baseY = (story - 1) * STORY_H;
      cornerOffsets.forEach(([cx, cz]) => {
        const colGeo = new THREE.CylinderGeometry(0.12, 0.12, colHeight, 14);
        const colMesh = new THREE.Mesh(colGeo, columnMat);
        colMesh.position.set(cx, baseY + colHeight / 2, cz);
        colMesh.castShadow = true;
        colMesh.receiveShadow = true;
        scene.add(colMesh);

        columns.push({
          mesh: colMesh,
          storyIdx: story,
          colX: cx,
          colZ: cz,
          height: colHeight,
          baseY,
        });
      });
    }

    slabsRef.current = slabs;
    columnMeshesRef.current = columns;
    nodeSpheresRef.current = nodeSpheres;

    // 9. Spatial Seismic Wave (Muted mint travelling wavefronts)
    const waveLines: THREE.Line[] = [];
    const WAVE_COUNT = 5;
    for (let i = 0; i < WAVE_COUNT; i++) {
      const pts: THREE.Vector3[] = [];
      const segs = 36;
      for (let s = 0; s <= segs; s++) {
        const angle = -Math.PI / 3 + (s / segs) * ((2 * Math.PI) / 3);
        pts.push(new THREE.Vector3(Math.cos(angle), 0.02, Math.sin(angle)));
      }
      const waveGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const waveMat = new THREE.LineBasicMaterial({
        color: 0x73e6b5,
        transparent: true,
        opacity: 0.12 + (i / WAVE_COUNT) * 0.3,
      });
      const line = new THREE.Line(waveGeo, waveMat);
      line.visible = false;
      scene.add(line);
      waveLines.push(line);
    }
    waveLinesRef.current = waveLines;

    // 10. Epicenter 3D Marker Group (Warm Amber / Rust)
    const epiGroup = new THREE.Group();
    epiGroup.visible = false;

    const epiRingGeo = new THREE.RingGeometry(0.28, 0.42, 24);
    const epiRingMat = new THREE.MeshBasicMaterial({ color: 0xd6b56d, side: THREE.DoubleSide });
    const epiRing = new THREE.Mesh(epiRingGeo, epiRingMat);
    epiRing.rotation.x = -Math.PI / 2;
    epiRing.position.y = 0.03;
    epiGroup.add(epiRing);

    const hypoPts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -2.4, 0)];
    const hypoGeo = new THREE.BufferGeometry().setFromPoints(hypoPts);
    const hypoMat = new THREE.LineDashedMaterial({
      color: 0xd6b56d,
      dashSize: 0.2,
      gapSize: 0.15,
      transparent: true,
      opacity: 0.75,
    });
    const hypoLine = new THREE.Line(hypoGeo, hypoMat);
    hypoLine.computeLineDistances();
    epiGroup.add(hypoLine);

    const hypoBead = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xd6b56d })
    );
    hypoBead.position.y = -2.4;
    epiGroup.add(hypoBead);

    scene.add(epiGroup);
    epicenterGroupRef.current = epiGroup;

    // Resize
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Animation Loop
    let clock = new THREE.Clock();
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Camera lerp
      if (targetCamPosRef.current && camera) {
        camera.position.lerp(targetCamPosRef.current, 0.06);
        if (camera.position.distanceTo(targetCamPosRef.current) < 0.05) {
          targetCamPosRef.current = null;
        }
      }

      controls.update();

      // Travelling wave propagation
      if (waveLinesRef.current.length > 0) {
        waveLinesRef.current.forEach((wLine, idx) => {
          const speed = 2.6;
          const cycle = (elapsedTime * speed + idx * 1.5) % 11.5;
          const radius = 2.0 + cycle;
          wLine.scale.set(radius, 1, radius);
          const mat = wLine.material as THREE.LineBasicMaterial;
          mat.opacity = Math.max(0, 0.45 * (1.0 - cycle / 11.5));
        });
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Epicenter when event changes
  useEffect(() => {
    if (!epicenterGroupRef.current || !waveLinesRef.current) return;

    if (!selectedEvent) {
      epicenterGroupRef.current.visible = false;
      waveLinesRef.current.forEach((w) => (w.visible = false));
      return;
    }

    epicenterGroupRef.current.visible = true;
    waveLinesRef.current.forEach((w) => (w.visible = true));

    const dLat = selectedEvent.latitude - 34.1377;
    const dLon = selectedEvent.longitude - (-118.1253);
    const azimuth = Math.atan2(dLon, dLat);

    const epiDist = 10.8;
    const ex = Math.sin(azimuth) * epiDist;
    const ez = Math.cos(azimuth) * epiDist;
    epicenterGroupRef.current.position.set(ex, 0, ez);

    waveLinesRef.current.forEach((w) => {
      w.position.set(ex, 0, ez);
      w.rotation.y = azimuth + Math.PI;
    });
  }, [selectedEvent]);

  // Structural dynamic response
  useEffect(() => {
    let animId: number;
    let localClock = new THREE.Clock();

    const updateStructuralMotion = () => {
      animId = requestAnimationFrame(updateStructuralMotion);
      const t = localClock.getElapsedTime();

      let baseAmpMm = 0.0;
      if (simResult && simResult.u_pred.length > 0) {
        const len = simResult.u_pred.length;
        const sampleIdx = Math.floor((t * 25) % len);
        baseAmpMm = simResult.u_pred[sampleIdx] * 1000.0;
      } else if (selectedEvent) {
        const distKm = selectedEvent.distance_km || 1000;
        const mag = selectedEvent.magnitude;
        const atten = Math.max(0.1, Math.min(2.5, Math.pow(10, 0.4 * (mag - 5.0)) / (1.0 + distKm / 400.0)));
        const omega1 = 2.0 * Math.PI * 1.8;
        const omega2 = 2.0 * Math.PI * 5.4;
        const sway = Math.sin(omega1 * t) * 0.85 + Math.sin(omega2 * t) * 0.15;
        baseAmpMm = atten * 11.5 * sway;
      }

      const scaledDrift3D = (baseAmpMm / 1000.0) * (visualScale * 0.08);

      setCurrentDriftMm(baseAmpMm);
      setPeakRoofDispMm((prev) => Math.max(prev, Math.abs(baseAmpMm)));

      if (slabsRef.current.length === STORY_COUNT) {
        const floorDrifts = [scaledDrift3D * 0.35, scaledDrift3D * 0.7, scaledDrift3D * 1.0];

        slabsRef.current.forEach((slab, idx) => {
          slab.position.x = floorDrifts[idx];
        });

        columnMeshesRef.current.forEach((col) => {
          const s = col.storyIdx - 1;
          const lowerX = s === 0 ? 0 : floorDrifts[s - 1];
          const upperX = floorDrifts[s];
          const midX = (lowerX + upperX) / 2;
          const dx = upperX - lowerX;

          col.mesh.position.x = col.colX + midX;
          col.mesh.rotation.z = -Math.atan2(dx, col.height);
        });
      }
    };

    updateStructuralMotion();

    return () => cancelAnimationFrame(animId);
  }, [selectedEvent, visualScale, simResult]);

  const handlePreset = useCallback((preset: "iso" | "front" | "top") => {
    setCameraPreset(preset);
    targetCamPosRef.current = CAM_PRESETS[preset].clone();
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 3.8, 0);
    }
  }, []);

  const handleReset = useCallback(() => {
    handlePreset("iso");
  }, [handlePreset]);

  return (
    <div className="relative w-full h-full min-h-[460px] lg:min-h-[500px] rounded-lg overflow-hidden select-none bg-[#0B1714]">
      {/* 3D WebGL Canvas */}
      <div ref={mountRef} className="w-full h-full min-h-[460px] lg:min-h-[500px]" />

      {/* WebGL Fallback */}
      {!hasWebGL && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0B1714] text-[#82928B] p-6 text-center">
          <AlertCircle size={28} className="text-[#D6B56D] mb-2" />
          <span className="font-sans text-xs font-semibold text-[#E8E8DE] tracking-wide">
            WebGL acceleration unavailable
          </span>
          <span className="font-mono text-[11px] text-[#82928B] mt-1">
            Displaying structural schematic: Roof drift u₃ = {currentDriftMm.toFixed(2)} mm
          </span>
        </div>
      )}

      {/* TOP-LEFT METADATA OVERLAY (Restrained, Editorial) */}
      <div className="absolute top-4 left-4 pointer-events-none space-y-1 font-sans">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
          <span className="text-xs font-semibold text-[#E8E8DE] tracking-tight">
            3-Story Benchmark Frame
          </span>
          <span className="text-[10px] font-mono text-[#82928B]">
            Pasadena Reference
          </span>
        </div>
        <div className="text-[11px] font-mono text-[#82928B] pl-3.5">
          Surrogate: <span className="text-[#73E6B5]">EXP6 Multi-Modal GNO</span>
          {isSimulating && <span className="ml-2 text-[#D6B56D] animate-pulse font-sans font-medium">Computing...</span>}
        </div>
      </div>

      {/* TOP-RIGHT VIEWPORT PRESET CONTROLS */}
      <div className="absolute top-4 right-4 flex items-center gap-1 z-10 font-mono text-[10px]">
        <div className="flex items-center bg-[#07110F]/80 border border-white/[0.06] p-0.5 rounded backdrop-blur-sm">
          <button
            onClick={() => handlePreset("iso")}
            className={`px-2 py-0.5 rounded transition cursor-pointer ${
              cameraPreset === "iso"
                ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            ISO
          </button>
          <button
            onClick={() => handlePreset("front")}
            className={`px-2 py-0.5 rounded transition cursor-pointer ${
              cameraPreset === "front"
                ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            FRONT
          </button>
          <button
            onClick={() => handlePreset("top")}
            className={`px-2 py-0.5 rounded transition cursor-pointer ${
              cameraPreset === "top"
                ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            TOP
          </button>
          <button
            onClick={handleReset}
            className="px-1.5 py-0.5 text-[#82928B] hover:text-[#E8E8DE] cursor-pointer border-l border-white/[0.06] ml-0.5"
            title="Reset Camera"
          >
            <RotateCcw size={10} />
          </button>
        </div>
      </div>

      {/* BOTTOM-LEFT REAL-TIME DRIFT HUD */}
      <div className="absolute bottom-4 left-4 pointer-events-none flex items-center gap-3 font-mono text-[11px]">
        <div className="px-2.5 py-1 bg-[#07110F]/80 border border-white/[0.06] rounded backdrop-blur-sm flex items-center gap-2">
          <span className="text-[#82928B]">Roof Drift:</span>
          <span className="text-[#73E6B5] font-semibold font-mono">
            {currentDriftMm >= 0 ? "+" : ""}{currentDriftMm.toFixed(2)} mm
          </span>
          <span className="text-[#82928B]">|</span>
          <span className="text-[#82928B]">Peak:</span>
          <span className="text-[#E8E8DE] font-mono">
            {peakRoofDispMm.toFixed(2)} mm
          </span>
        </div>
      </div>

      {/* BOTTOM-RIGHT SCALE MULTIPLIER */}
      <div className="absolute bottom-4 right-4 flex items-center gap-1.5 z-10 font-mono text-[10px]">
        <div className="flex items-center bg-[#07110F]/80 border border-white/[0.06] px-2 py-0.5 rounded backdrop-blur-sm text-[#82928B]">
          <span className="mr-1 text-[#82928B]">Scale:</span>
          {[5, 10, 20].map((s) => (
            <button
              key={s}
              onClick={() => onChangeVisualScale(s)}
              className={`px-1.5 py-0.5 rounded cursor-pointer transition ${
                visualScale === s
                  ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                  : "hover:text-[#E8E8DE]"
              }`}
            >
              {s}×
            </button>
          ))}
          <span className="text-[9px] text-[#82928B] ml-1 hidden sm:inline">visual</span>
        </div>
      </div>
    </div>
  );
};
