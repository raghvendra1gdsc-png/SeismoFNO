import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RotateCcw, AlertCircle, Play } from "lucide-react";
import type { LiveEarthquakeEvent, DemoSimulationResponse } from "../../api/researchDemoApi";

interface LiveSeismic3DViewportProps {
  selectedEvent: LiveEarthquakeEvent | null;
  visualScale: number;
  onChangeVisualScale: (scale: number) => void;
  isSimulating: boolean;
  simResult: DemoSimulationResponse | null;
  storyCount?: number;
  triggerTransientToken?: number;
}

export const LiveSeismic3DViewport: React.FC<LiveSeismic3DViewportProps> = ({
  selectedEvent,
  visualScale,
  onChangeVisualScale,
  isSimulating,
  simResult,
  storyCount = 3,
  triggerTransientToken,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  const [hasWebGL, setHasWebGL] = useState<boolean>(true);
  const [cameraPreset, setCameraPreset] = useState<"iso" | "front" | "top">("iso");
  const [viewMode, setViewMode] = useState<"peak" | "transient" | "mode1">("transient");

  const roofDriftSpanRef = useRef<HTMLSpanElement>(null);
  const statusSpanRef = useRef<HTMLSpanElement>(null);
  const transientClockRef = useRef<number>(0);

  // Single canonical peak displacement from surrogate forward pass
  const peakMm = useMemo(() => {
    if (simResult?.metrics?.peak_pred_m !== undefined && simResult.metrics.peak_pred_m !== null) {
      return simResult.metrics.peak_pred_m * 1000.0;
    }
    if (simResult && simResult.u_pred.length > 0) {
      return Math.max(...simResult.u_pred.map(Math.abs)) * 1000.0;
    }
    return 0.39;
  }, [simResult]);

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
  const buildingGroupRef = useRef<THREE.Group | null>(null);
  const buildStructureRef = useRef<((n: number) => void) | null>(null);

  // Camera Presets (approx x=7, y=5, z=8 looking toward model)
  const CAM_PRESETS = {
    iso: new THREE.Vector3(7.2, 5.0, 8.2),
    front: new THREE.Vector3(0.0, 4.4, 12.0),
    top: new THREE.Vector3(0.0, 15.5, 0.001),
  };

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

    // 1. Scene with clean engineering off-white canvas
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xF8FAFC);
    scene.fog = new THREE.FogExp2(0xF8FAFC, 0.015);
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

    // 5. Studio Lighting (Clean White Key + Soft Ambient + Studio Rim)
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.15);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
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

    const fillLight = new THREE.DirectionalLight(0xCBD5E1, 0.4);
    fillLight.position.set(-9, 7, -9);
    scene.add(fillLight);

    // 6. Ground Plane & Subtle Engineering Grid
    const groundGeo = new THREE.PlaneGeometry(36, 36);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0xF8FAFC,
      roughness: 0.95,
      metalness: 0.05,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);

    const gridHelper = new THREE.GridHelper(30, 30, 0xCBD5E1, 0xE2E8F0);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    // 7. Foundation Footings
    const footingGeo = new THREE.BoxGeometry(0.85, 0.28, 0.85);
    const footingMat = new THREE.MeshStandardMaterial({
      color: 0xE2E8F0,
      roughness: 0.8,
      metalness: 0.1,
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

    // 8. Materials for Structural Frame
    const slabMat = new THREE.MeshStandardMaterial({
      color: 0xFFFFFF,
      roughness: 0.4,
      metalness: 0.1,
    });
    const slabEdgeMat = new THREE.LineBasicMaterial({ color: 0x94A3B8, transparent: true, opacity: 0.8 });
    const columnMat = new THREE.MeshStandardMaterial({
      color: 0x334155, // Graphite slate
      roughness: 0.35,
      metalness: 0.2,
    });
    const beamMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.4,
      metalness: 0.2,
    });
    const nodeMat = new THREE.MeshStandardMaterial({
      color: 0x047857,
      emissive: 0x047857,
      emissiveIntensity: 0.2,
      roughness: 0.25,
    });

    const nodeGeo = new THREE.SphereGeometry(0.12, 16, 16);

    // 8. Structural Multi-Story Frame Container
    const buildingGroup = new THREE.Group();
    scene.add(buildingGroup);
    buildingGroupRef.current = buildingGroup;

    const buildStructure = (numStories: number) => {
      if (!buildingGroupRef.current) return;
      const group = buildingGroupRef.current;
      while (group.children.length > 0) {
        group.remove(group.children[0]);
      }
      slabsRef.current = [];
      columnMeshesRef.current = [];
      nodeSpheresRef.current = [];

      const slabs: THREE.Group[] = [];
      const columns: typeof columnMeshesRef.current = [];
      const nodeSpheres: THREE.Mesh[] = [];

      for (let story = 1; story <= numStories; story++) {
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

        group.add(floorGroup);
        slabs.push(floorGroup);

        // 4 Columns Below This Floor
        const colHeight = STORY_H;
        const baseY = (story - 1) * STORY_H;
        cornerOffsets.forEach(([cx, cz]) => {
          const colGeo = new THREE.CylinderGeometry(0.12, 0.12, colHeight, 14);
          const colMesh = new THREE.Mesh(colGeo, columnMat);
          colMesh.position.set(cx, baseY + colHeight / 2, cz);
          colMesh.castShadow = true;
          colMesh.receiveShadow = true;
          group.add(colMesh);

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

      if (controlsRef.current) {
        controlsRef.current.target.set(0, (numStories * STORY_H) * 0.45, 0);
      }
    };

    buildStructureRef.current = buildStructure;
    buildStructure(storyCount || 3);

    // 9. Spatial Seismic Wave (Muted emerald travelling wavefronts)
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
        color: 0x047857,
        transparent: true,
        opacity: 0.12 + (i / WAVE_COUNT) * 0.25,
      });
      const line = new THREE.Line(waveGeo, waveMat);
      line.visible = false;
      scene.add(line);
      waveLines.push(line);
    }
    waveLinesRef.current = waveLines;

    // 10. Epicenter 3D Marker Group (Amber Engineering Beacon)
    const epiGroup = new THREE.Group();
    epiGroup.visible = false;

    const epiRingGeo = new THREE.RingGeometry(0.28, 0.42, 24);
    const epiRingMat = new THREE.MeshBasicMaterial({ color: 0xB45309, side: THREE.DoubleSide });
    const epiRing = new THREE.Mesh(epiRingGeo, epiRingMat);
    epiRing.rotation.x = -Math.PI / 2;
    epiRing.position.y = 0.03;
    epiGroup.add(epiRing);

    const hypoPts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -2.4, 0)];
    const hypoGeo = new THREE.BufferGeometry().setFromPoints(hypoPts);
    const hypoMat = new THREE.LineDashedMaterial({
      color: 0xB45309,
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
      new THREE.MeshBasicMaterial({ color: 0xB45309 })
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
          mat.opacity = Math.max(0, 0.4 * (1.0 - cycle / 11.5));
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
    let lastTime = performance.now();

    const updateStructuralMotion = () => {
      animId = requestAnimationFrame(updateStructuralMotion);
      const now = performance.now();
      const deltaSec = Math.min((now - lastTime) / 1000.0, 0.1);
      lastTime = now;

      let baseAmpMm = 0.0;

      if (viewMode === "peak") {
        baseAmpMm = peakMm;
        if (roofDriftSpanRef.current) {
          roofDriftSpanRef.current.textContent = `+${peakMm.toFixed(2)} mm`;
        }
        if (statusSpanRef.current) {
          statusSpanRef.current.textContent = "Settled (Peak Envelope)";
        }
      } else if (viewMode === "mode1") {
        // Continuous harmonic modal oscillation at fundamental frequency
        const t = performance.now() / 1000.0;
        const omega1 = 2.0 * Math.PI / 0.55;
        baseAmpMm = peakMm * Math.sin(t * omega1);
        if (roofDriftSpanRef.current) {
          roofDriftSpanRef.current.textContent = `${baseAmpMm >= 0 ? "+" : ""}${baseAmpMm.toFixed(2)} mm`;
        }
        if (statusSpanRef.current) {
          statusSpanRef.current.textContent = "Harmonic (Mode 1 Invariant)";
        }
      } else {
        // Dynamic Transient mode (Vivid time-history vibration)
        if (simResult && simResult.u_pred && simResult.u_pred.length > 0) {
          const len = simResult.u_pred.length;
          const dt = simResult.earthquake?.dt || 0.02;
          const totalDuration = len * dt;

          // Advance at 1.35x responsive speed so shaking is immediately perceived
          transientClockRef.current += deltaSec * 1.35;
          const t = transientClockRef.current;

          if (t < totalDuration) {
            const sampleIdx = Math.min(len - 1, Math.floor(t / dt));
            baseAmpMm = simResult.u_pred[sampleIdx] * 1000.0;
            if (roofDriftSpanRef.current) {
              roofDriftSpanRef.current.textContent = `${baseAmpMm >= 0 ? "+" : ""}${baseAmpMm.toFixed(2)} mm`;
            }
            if (statusSpanRef.current) {
              statusSpanRef.current.textContent = `Vibrating (t = ${t.toFixed(1)}s / ${totalDuration.toFixed(1)}s)`;
            }
          } else {
            // Reached record terminus: settle at residual offset
            baseAmpMm = simResult.u_pred[len - 1] * 1000.0;
            if (roofDriftSpanRef.current) {
              roofDriftSpanRef.current.textContent = `${baseAmpMm >= 0 ? "+" : ""}${baseAmpMm.toFixed(2)} mm`;
            }
            if (statusSpanRef.current) {
              statusSpanRef.current.textContent = `Settled at Residual (${baseAmpMm >= 0 ? "+" : ""}${baseAmpMm.toFixed(2)} mm)`;
            }
          }
        } else {
          baseAmpMm = peakMm;
          if (roofDriftSpanRef.current) {
            roofDriftSpanRef.current.textContent = `+${peakMm.toFixed(2)} mm`;
          }
          if (statusSpanRef.current) {
            statusSpanRef.current.textContent = "Settled (Peak Envelope)";
          }
        }
      }

      // Visual amplification factor:
      // Real structural drifts are millimeters. To make dynamic motion vividly visible in 3D:
      // Default scale (10x) maps 1mm to 0.45 3D units, ensuring crisp, realistic vibration.
      const vScale = visualScale || 10;
      const scaleMultiplier = (vScale / 10.0) * 0.45;
      const scaledDrift3D = Math.sign(baseAmpMm) * Math.min(Math.abs(baseAmpMm) * scaleMultiplier, 3.8);

      const nStories = slabsRef.current.length;
      if (nStories > 0) {
        const floorDrifts = Array.from({ length: nStories }, (_, idx) => {
          const ratio = (idx + 1) / nStories;
          return scaledDrift3D * ratio;
        });

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
  }, [selectedEvent, visualScale, simResult, viewMode, peakMm]);

  // Rebuild building whenever storyCount prop changes
  useEffect(() => {
    if (buildStructureRef.current) {
      buildStructureRef.current(storyCount);
    }
  }, [storyCount]);

  // Trigger transient playback whenever triggerTransientToken or new simResult arrives
  useEffect(() => {
    if (triggerTransientToken) {
      setViewMode("transient");
      transientClockRef.current = 0;
    }
  }, [triggerTransientToken]);

  useEffect(() => {
    if (simResult) {
      setViewMode("transient");
      transientClockRef.current = 0;
    }
  }, [simResult]);

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
    <div className="relative w-full h-full min-h-[460px] lg:min-h-[500px] rounded-lg overflow-hidden select-none bg-[#F8FAFC]">
      {/* 3D WebGL Canvas */}
      <div ref={mountRef} className="w-full h-full min-h-[460px] lg:min-h-[500px]" />

      {/* WebGL Fallback */}
      {!hasWebGL && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#F8FAFC] text-[#64748B] p-6 text-center">
          <AlertCircle size={28} className="text-[#B45309] mb-2" />
          <span className="font-sans text-xs font-semibold text-[#0F172A] tracking-wide">
            WebGL acceleration unavailable
          </span>
          <span className="font-mono text-[11px] text-[#64748B] mt-1">
            Displaying structural schematic: Roof drift u₃ = {peakMm.toFixed(2)} mm
          </span>
        </div>
      )}

      {/* TOP-LEFT METADATA OVERLAY */}
      <div className="absolute top-4 left-4 pointer-events-none space-y-1 font-sans">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#047857] animate-pulse" />
          <span className="text-xs font-bold text-[#0F172A] tracking-tight">
            {storyCount}-Story Structural Frame
          </span>
          <span className="badge-tech bg-white border border-[#CBD5E1] text-[#334155]">
            Physical Model
          </span>
        </div>
        <div className="text-[11px] font-mono text-[#64748B] pl-3.5">
          Surrogate: <span className="text-[#047857] font-semibold">EXP6 Multi-Modal GNO</span>
          {isSimulating && <span className="ml-2 text-[#047857] font-bold font-mono animate-pulse">Computing...</span>}
        </div>
      </div>

      {/* TOP-RIGHT VIEWPORT PRESET & STATE CONTROLS */}
      <div className="absolute top-4 right-4 flex flex-wrap items-center gap-1.5 z-10 font-mono text-[10px]">
        {/* VIEW MODE: PEAK STATE (FIXED) vs DYNAMIC TRANSIENT vs MODE 1 */}
        <div className="flex items-center bg-white/95 border border-[#CBD5E1] p-0.5 rounded shadow-xs">
          <button
            onClick={() => setViewMode("peak")}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold ${
              viewMode === "peak"
                ? "bg-[#047857] text-white"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
            title="Display maximum peak drift envelope (steady single value)"
          >
            PEAK STATE
          </button>
          <button
            onClick={() => {
              setViewMode("transient");
              transientClockRef.current = 0;
            }}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold flex items-center gap-1 ${
              viewMode === "transient"
                ? "bg-[#047857] text-white"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
            title="Play time-history motion and settle at end of earthquake"
          >
            <Play size={9} className={viewMode === "transient" ? "fill-white" : ""} />
            TRANSIENT
          </button>
          <button
            onClick={() => setViewMode("mode1")}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold ${
              viewMode === "mode1"
                ? "bg-[#047857] text-white"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
            title="Display harmonic Mode 1 invariant oscillation"
          >
            MODE 1
          </button>
          <button
            onClick={() => {
              setViewMode("transient");
              transientClockRef.current = 0;
            }}
            className="px-1.5 py-0.5 text-[#64748B] hover:text-[#0F172A] cursor-pointer border-l border-[#E2E8F0] ml-0.5"
            title="Replay Waveform Vibration"
          >
            <RotateCcw size={10} />
          </button>
        </div>

        {/* CAMERA PRESETS */}
        <div className="flex items-center bg-white/95 border border-[#CBD5E1] p-0.5 rounded shadow-xs">
          <button
            onClick={() => handlePreset("iso")}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-medium ${
              cameraPreset === "iso"
                ? "bg-[#047857] text-white font-semibold"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            ISO
          </button>
          <button
            onClick={() => handlePreset("front")}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-medium ${
              cameraPreset === "front"
                ? "bg-[#047857] text-white font-semibold"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            FRONT
          </button>
          <button
            onClick={() => handlePreset("top")}
            className={`px-2 py-0.5 rounded transition cursor-pointer font-medium ${
              cameraPreset === "top"
                ? "bg-[#047857] text-white font-semibold"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            TOP
          </button>
          <button
            onClick={handleReset}
            className="px-1.5 py-0.5 text-[#64748B] hover:text-[#0F172A] cursor-pointer border-l border-[#E2E8F0] ml-0.5"
            title="Reset Camera"
          >
            <RotateCcw size={10} />
          </button>
        </div>
      </div>

      {/* BOTTOM-LEFT REAL-TIME DRIFT HUD */}
      <div className="absolute bottom-4 left-4 pointer-events-none flex items-center gap-3 font-mono text-[11px]">
        <div className="px-3 py-1 bg-white/95 border border-[#CBD5E1] rounded shadow-xs flex items-center gap-2">
          <span className="text-[#64748B]">Roof Drift:</span>
          <span ref={roofDriftSpanRef} className="text-[#047857] font-semibold font-mono">
            +{peakMm.toFixed(2)} mm
          </span>
          <span className="text-[#CBD5E1]">|</span>
          <span className="text-[#64748B]">Peak:</span>
          <span className="text-[#0F172A] font-bold font-mono">
            {peakMm.toFixed(2)} mm
          </span>
          <span className="text-[#CBD5E1]">|</span>
          <span ref={statusSpanRef} className="text-[10px] text-[#475569] font-mono uppercase font-semibold">
            Settled (Peak State)
          </span>
        </div>
      </div>

      {/* BOTTOM-RIGHT SCALE MULTIPLIER */}
      <div className="absolute bottom-4 right-4 flex items-center gap-1.5 z-10 font-mono text-[10px]">
        <div className="flex items-center bg-white/95 border border-[#CBD5E1] px-2 py-0.5 rounded shadow-xs text-[#64748B]">
          <span className="mr-1 text-[#64748B] font-semibold">Scale:</span>
          {[5, 10, 20].map((s) => (
            <button
              key={s}
              onClick={() => onChangeVisualScale(s)}
              className={`px-1.5 py-0.5 rounded cursor-pointer transition ${
                visualScale === s
                  ? "bg-[#047857] text-white font-semibold"
                  : "hover:text-[#0F172A]"
              }`}
            >
              {s}×
            </button>
          ))}
          <span className="text-[9px] text-[#94A3B8] ml-1 hidden sm:inline">visual</span>
        </div>
      </div>
    </div>
  );
};

