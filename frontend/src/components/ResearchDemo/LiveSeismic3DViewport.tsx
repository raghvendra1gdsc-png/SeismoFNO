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

  // Structural Model References
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

  // Target camera presets (approx x=7, y=5, z=8 looking toward model)
  const CAM_PRESETS = {
    iso: new THREE.Vector3(7.0, 5.2, 8.0),
    front: new THREE.Vector3(0.0, 4.5, 12.5),
    top: new THREE.Vector3(0.0, 16.0, 0.001),
  };

  // Dimensions
  const STORY_COUNT = 3;
  const STORY_H = 2.8;
  const BAY_W = 5.6;
  const BAY_D = 5.6;

  // Initialize WebGL Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Check WebGL support
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

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 520;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c12);
    scene.fog = new THREE.FogExp2(0x080c12, 0.022);
    sceneRef.current = scene;

    // 2. Camera setup (approx x=7, y=5, z=8 looking toward model)
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.copy(CAM_PRESETS.iso);
    cameraRef.current = camera;

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 3.8, 0);
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // Keep above ground plane
    controls.minDistance = 4.0;
    controls.maxDistance = 28.0;
    controlsRef.current = controls;

    // 5. Restrained Lighting (Key + Subtle Fill)
    const ambientLight = new THREE.AmbientLight(0x151d27, 0.9);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xe8edf3, 1.1);
    keyLight.position.set(8, 14, 10);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x28d7ff, 0.35);
    fillLight.position.set(-8, 6, -8);
    scene.add(fillLight);

    // 6. Engineering Ground Plane & Grid
    const gridHelper = new THREE.GridHelper(30, 30, 0x222e3e, 0x111821);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Subtle dark circular platform
    const platformGeo = new THREE.CircleGeometry(14, 48);
    const platformMat = new THREE.MeshBasicMaterial({ color: 0x0a0f17, depthWrite: false });
    const platform = new THREE.Mesh(platformGeo, platformMat);
    platform.rotation.x = -Math.PI / 2;
    platform.position.y = -0.02;
    scene.add(platform);

    // 7. Foundation Footings
    const footingGeo = new THREE.BoxGeometry(0.8, 0.25, 0.8);
    const footingMat = new THREE.MeshStandardMaterial({ color: 0x151d27, roughness: 0.7, metalness: 0.2 });
    const cornerOffsets = [
      [-BAY_W / 2, -BAY_D / 2],
      [BAY_W / 2, -BAY_D / 2],
      [-BAY_W / 2, BAY_D / 2],
      [BAY_W / 2, BAY_D / 2],
    ];

    cornerOffsets.forEach(([fx, fz]) => {
      const footing = new THREE.Mesh(footingGeo, footingMat);
      footing.position.set(fx, 0.125, fz);
      scene.add(footing);
    });

    // 8. Multi-Story Structural Frame (3-Story RC Frame)
    const slabs: THREE.Group[] = [];
    const columns: typeof columnMeshesRef.current = [];
    const nodeSpheres: THREE.Mesh[] = [];

    // Materials
    const slabMat = new THREE.MeshStandardMaterial({
      color: 0x111821,
      roughness: 0.6,
      metalness: 0.3,
      transparent: true,
      opacity: 0.85,
    });
    const slabEdgeMat = new THREE.LineBasicMaterial({ color: 0x28d7ff, transparent: true, opacity: 0.35 });
    const columnMat = new THREE.MeshStandardMaterial({
      color: 0x263342,
      roughness: 0.45,
      metalness: 0.35,
    });
    const beamMat = new THREE.MeshStandardMaterial({
      color: 0x1d2734,
      roughness: 0.5,
      metalness: 0.3,
    });
    const nodeMat = new THREE.MeshStandardMaterial({
      color: 0x28d7ff,
      emissive: 0x28d7ff,
      emissiveIntensity: 0.4,
      roughness: 0.2,
    });

    const nodeGeo = new THREE.SphereGeometry(0.12, 16, 16);

    for (let story = 1; story <= STORY_COUNT; story++) {
      const floorY = story * STORY_H;
      const floorGroup = new THREE.Group();
      floorGroup.position.set(0, floorY, 0);

      // Floor Slab
      const slabMesh = new THREE.Mesh(new THREE.BoxGeometry(BAY_W + 0.5, 0.18, BAY_D + 0.5), slabMat);
      floorGroup.add(slabMesh);

      // Slab Wireframe Outline
      const slabEdges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(BAY_W + 0.5, 0.18, BAY_D + 0.5)),
        slabEdgeMat
      );
      floorGroup.add(slabEdges);

      // Perimeter Beams
      const beamGeoX = new THREE.BoxGeometry(BAY_W, 0.2, 0.2);
      const beamGeoZ = new THREE.BoxGeometry(0.2, 0.2, BAY_D);

      const b1 = new THREE.Mesh(beamGeoX, beamMat);
      b1.position.set(0, -0.1, -BAY_D / 2);
      floorGroup.add(b1);

      const b2 = new THREE.Mesh(beamGeoX, beamMat);
      b2.position.set(0, -0.1, BAY_D / 2);
      floorGroup.add(b2);

      const b3 = new THREE.Mesh(beamGeoZ, beamMat);
      b3.position.set(-BAY_W / 2, -0.1, 0);
      floorGroup.add(b3);

      const b4 = new THREE.Mesh(beamGeoZ, beamMat);
      b4.position.set(BAY_W / 2, -0.1, 0);
      floorGroup.add(b4);

      // Structural Nodes at Column-Beam Intersections
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
        const colGeo = new THREE.CylinderGeometry(0.11, 0.11, colHeight, 12);
        const colMesh = new THREE.Mesh(colGeo, columnMat);
        colMesh.position.set(cx, baseY + colHeight / 2, cz);
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

    // 9. Travelling Seismic Spatial Wave (Concentric arcs on ground plane)
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
        color: 0x28d7ff,
        transparent: true,
        opacity: 0.15 + (i / WAVE_COUNT) * 0.35,
      });
      const line = new THREE.Line(waveGeo, waveMat);
      line.visible = false;
      scene.add(line);
      waveLines.push(line);
    }
    waveLinesRef.current = waveLines;

    // 10. Epicenter 3D Marker Group
    const epiGroup = new THREE.Group();
    epiGroup.visible = false;

    // Ground circle pulse
    const epiRingGeo = new THREE.RingGeometry(0.3, 0.45, 24);
    const epiRingMat = new THREE.MeshBasicMaterial({ color: 0xe35d5d, side: THREE.DoubleSide });
    const epiRing = new THREE.Mesh(epiRingGeo, epiRingMat);
    epiRing.rotation.x = -Math.PI / 2;
    epiRing.position.y = 0.03;
    epiGroup.add(epiRing);

    // Vertical hypocenter dashed line down to subsurface
    const hypoPts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -2.5, 0)];
    const hypoGeo = new THREE.BufferGeometry().setFromPoints(hypoPts);
    const hypoMat = new THREE.LineDashedMaterial({
      color: 0xe35d5d,
      dashSize: 0.2,
      gapSize: 0.15,
      transparent: true,
      opacity: 0.7,
    });
    const hypoLine = new THREE.Line(hypoGeo, hypoMat);
    hypoLine.computeLineDistances();
    epiGroup.add(hypoLine);

    // Subsurface hypocenter bead
    const hypoBead = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xe35d5d })
    );
    hypoBead.position.y = -2.5;
    epiGroup.add(hypoBead);

    scene.add(epiGroup);
    epicenterGroupRef.current = epiGroup;

    // Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // 11. Animation Loop
    let clock = new THREE.Clock();
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Smooth camera preset lerp
      if (targetCamPosRef.current && camera) {
        camera.position.lerp(targetCamPosRef.current, 0.06);
        if (camera.position.distanceTo(targetCamPosRef.current) < 0.05) {
          targetCamPosRef.current = null;
        }
      }

      controls.update();

      // Update travelling wave animation
      if (waveLinesRef.current.length > 0) {
        waveLinesRef.current.forEach((wLine, idx) => {
          const speed = 2.8;
          const cycle = (elapsedTime * speed + idx * 1.6) % 12.0;
          const radius = 2.0 + cycle;
          wLine.scale.set(radius, 1, radius);
          const mat = wLine.material as THREE.LineBasicMaterial;
          mat.opacity = Math.max(0, 0.55 * (1.0 - cycle / 12.0));
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

  // Update Epicenter & Wave Direction when selectedEvent changes
  useEffect(() => {
    if (!epicenterGroupRef.current || !waveLinesRef.current) return;

    if (!selectedEvent) {
      epicenterGroupRef.current.visible = false;
      waveLinesRef.current.forEach((w) => (w.visible = false));
      return;
    }

    epicenterGroupRef.current.visible = true;
    waveLinesRef.current.forEach((w) => (w.visible = true));

    // Calculate relative azimuth angle from Scenario Target (Pasadena: lat 34.14, lon -118.13)
    const dLat = selectedEvent.latitude - 34.1377;
    const dLon = selectedEvent.longitude - (-118.1253);
    const azimuth = Math.atan2(dLon, dLat); // angle in radians

    // Position Epicenter marker at distance ~11.0 in relative direction
    const epiDist = 11.0;
    const ex = Math.sin(azimuth) * epiDist;
    const ez = Math.cos(azimuth) * epiDist;
    epicenterGroupRef.current.position.set(ex, 0, ez);

    // Rotate wave arcs to propagate FROM epicenter TOWARD center
    waveLinesRef.current.forEach((w) => {
      w.position.set(ex, 0, ez);
      w.rotation.y = azimuth + Math.PI; // point back toward origin
    });
  }, [selectedEvent]);

  // Animate Structural Response
  useEffect(() => {
    let animId: number;
    let localClock = new THREE.Clock();

    const updateStructuralMotion = () => {
      animId = requestAnimationFrame(updateStructuralMotion);
      const t = localClock.getElapsedTime();

      // Determine dynamic displacement
      let baseAmpMm = 0.0;
      if (simResult && simResult.u_pred.length > 0) {
        // Read from real surrogate simulation waveform
        const len = simResult.u_pred.length;
        const sampleIdx = Math.floor((t * 25) % len);
        baseAmpMm = simResult.u_pred[sampleIdx] * 1000.0; // convert to mm
      } else if (selectedEvent) {
        // Physics-grounded visual modal sway driven by event magnitude & distance
        const distKm = selectedEvent.distance_km || 1000;
        const mag = selectedEvent.magnitude;
        // Attenuation scaling factor
        const atten = Math.max(0.1, Math.min(2.5, Math.pow(10, 0.4 * (mag - 5.0)) / (1.0 + distKm / 400.0)));
        // 3-mode synthetic vibration
        const omega1 = 2.0 * Math.PI * 1.8; // ~1.8 Hz fundamental frequency
        const omega2 = 2.0 * Math.PI * 5.4;
        const sway = Math.sin(omega1 * t) * 0.85 + Math.sin(omega2 * t) * 0.15;
        baseAmpMm = atten * 12.0 * sway;
      }

      const scaledDrift3D = (baseAmpMm / 1000.0) * (visualScale * 0.08); // 3D units

      setCurrentDriftMm(baseAmpMm);
      setPeakRoofDispMm((prev) => Math.max(prev, Math.abs(baseAmpMm)));

      // Apply first-mode shear deformation to floor slabs
      if (slabsRef.current.length === STORY_COUNT) {
        const floorDrifts = [scaledDrift3D * 0.35, scaledDrift3D * 0.7, scaledDrift3D * 1.0];

        slabsRef.current.forEach((slab, idx) => {
          slab.position.x = floorDrifts[idx];
        });

        // Update column segment rotations to connect sheared floors
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

  // Camera Presets Click Handlers
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
    <div className="relative w-full h-full min-h-[480px] lg:min-h-[520px] bg-[#080C12] border border-white/[0.07] rounded-[4px] overflow-hidden select-none">
      {/* 3D WebGL Canvas Container */}
      <div ref={mountRef} className="w-full h-full min-h-[480px] lg:min-h-[520px]" />

      {/* WebGL Fallback (if unavailable) */}
      {!hasWebGL && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#080C12] text-[#8D9AAA] p-6 text-center">
          <AlertCircle size={32} className="text-[#E8A63B] mb-2" />
          <span className="font-mono text-xs font-bold text-[#E8EDF3] uppercase tracking-wider">
            WEBGL HARDWARE ACCELERATION UNAVAILABLE
          </span>
          <span className="font-mono text-[11px] text-[#667487] mt-1 max-w-sm">
            3D VIEW FALLBACK ACTIVE · DISPLAYING MATHEMATICAL SCHEMATIC
          </span>
          <div className="mt-4 p-4 border border-white/[0.07] bg-[#0B1018] rounded-[3px] font-mono text-[11px] text-left space-y-1">
            <div>STORY 03: ROOF DRIFT u₃ = {currentDriftMm.toFixed(2)} mm</div>
            <div>STORY 02: DRIFT u₂ = {(currentDriftMm * 0.7).toFixed(2)} mm</div>
            <div>STORY 01: DRIFT u₁ = {(currentDriftMm * 0.35).toFixed(2)} mm</div>
            <div>SURROGATE: EXP6 MULTI-MODAL GNO</div>
          </div>
        </div>
      )}

      {/* TOP-LEFT HUD OVERLAY */}
      <div className="absolute top-3 left-3 pointer-events-none space-y-1">
        <div className="flex items-center gap-1.5 px-2 py-1 bg-[#0B1018]/90 border border-white/[0.07] rounded-[3px] backdrop-blur-md">
          <div className="w-1.5 h-1.5 rounded-full bg-[#28D7FF] shadow-[0_0_6px_#28D7FF]" />
          <span className="font-mono text-[11px] font-bold text-[#E8EDF3] tracking-wide uppercase">
            SEISMOFNO DIGITAL TWIN
          </span>
        </div>
        <div className="px-2 py-1 bg-[#0B1018]/80 border border-white/[0.07] rounded-[3px] font-mono text-[10px] text-[#8D9AAA]">
          MODEL: <span className="text-[#E8EDF3] font-semibold">3-STORY / 1-BAY RC FRAME</span>
        </div>
        <div className="px-2 py-1 bg-[#0B1018]/80 border border-white/[0.07] rounded-[3px] font-mono text-[10px] text-[#8D9AAA] flex items-center justify-between gap-2">
          <span>SOLVER: <span className="text-[#31D17C] font-semibold">EXP6 MULTI-MODAL GNO</span></span>
          {isSimulating && (
            <span className="text-[#28D7FF] font-bold animate-pulse">COMPUTING...</span>
          )}
        </div>
      </div>

      {/* TOP-RIGHT VIEWPORT CONTROLS */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 z-10">
        <div className="flex items-center bg-[#0B1018]/90 border border-white/[0.07] p-1 rounded-[3px] backdrop-blur-md text-[10px] font-mono">
          <button
            onClick={() => handlePreset("iso")}
            className={`px-2 py-0.5 rounded-[2px] cursor-pointer transition ${
              cameraPreset === "iso"
                ? "bg-[#28D7FF]/20 text-[#28D7FF] font-bold border border-[#28D7FF]/40"
                : "text-[#8D9AAA] hover:text-[#E8EDF3]"
            }`}
          >
            ISO
          </button>
          <button
            onClick={() => handlePreset("front")}
            className={`px-2 py-0.5 rounded-[2px] cursor-pointer transition ${
              cameraPreset === "front"
                ? "bg-[#28D7FF]/20 text-[#28D7FF] font-bold border border-[#28D7FF]/40"
                : "text-[#8D9AAA] hover:text-[#E8EDF3]"
            }`}
          >
            FRONT
          </button>
          <button
            onClick={() => handlePreset("top")}
            className={`px-2 py-0.5 rounded-[2px] cursor-pointer transition ${
              cameraPreset === "top"
                ? "bg-[#28D7FF]/20 text-[#28D7FF] font-bold border border-[#28D7FF]/40"
                : "text-[#8D9AAA] hover:text-[#E8EDF3]"
            }`}
          >
            TOP
          </button>
          <button
            onClick={handleReset}
            className="px-2 py-0.5 text-[#8D9AAA] hover:text-[#E8EDF3] cursor-pointer flex items-center gap-1 border-l border-white/[0.07] ml-1"
            title="Reset Viewport Camera"
          >
            <RotateCcw size={10} />
            <span>RESET</span>
          </button>
        </div>
      </div>

      {/* BOTTOM-LEFT HUD OVERLAY */}
      <div className="absolute bottom-3 left-3 pointer-events-none space-y-1">
        {selectedEvent ? (
          <div className="px-2.5 py-1.5 bg-[#0B1018]/90 border border-white/[0.07] rounded-[3px] backdrop-blur-md font-mono text-[10px] space-y-0.5">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#E35D5D] shadow-[0_0_6px_#E35D5D]" />
              <span className="text-[#8D9AAA]">OBSERVED EVENT:</span>
              <span className="text-[#E8EDF3] font-bold truncate max-w-[220px]">
                {selectedEvent.location}
              </span>
            </div>
            <div className="flex items-center gap-3 text-[#8D9AAA] pt-0.5">
              <span>
                MAG: <strong className="text-[#E35D5D]">M{selectedEvent.magnitude.toFixed(1)} Mw</strong>
              </span>
              <span>
                DEPTH: <strong className="text-[#E8EDF3]">{selectedEvent.depth_km.toFixed(1)} km</strong>
              </span>
              <span>
                DIST: <strong className="text-[#28D7FF]">{selectedEvent.distance_km ? `${selectedEvent.distance_km} km` : "N/A"}</strong>
              </span>
            </div>
          </div>
        ) : (
          <div className="px-2.5 py-1 bg-[#0B1018]/80 border border-white/[0.07] rounded-[3px] font-mono text-[10px] text-[#667487]">
            NO EVENT SELECTED · AWAITING USGS OBSERVATION
          </div>
        )}
      </div>

      {/* BOTTOM-RIGHT VISUAL SCALE & TELEMETRY CONTROLS */}
      <div className="absolute bottom-3 right-3 flex items-center gap-2 z-10">
        {/* Real-Time Story Node Telemetry */}
        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 bg-[#0B1018]/90 border border-white/[0.07] rounded-[3px] font-mono text-[10px] text-[#8D9AAA] backdrop-blur-md">
          <span>ROOF DRIFT:</span>
          <span className="text-[#28D7FF] font-bold font-mono">
            {currentDriftMm >= 0 ? "+" : ""}{currentDriftMm.toFixed(2)} mm
          </span>
          <span className="text-[#667487]">|</span>
          <span>PEAK:</span>
          <span className="text-[#E8EDF3] font-bold font-mono">
            {peakRoofDispMm.toFixed(2)} mm
          </span>
        </div>

        {/* Visual Scale Multiplier Selector */}
        <div className="flex items-center bg-[#0B1018]/90 border border-white/[0.07] px-2 py-1 rounded-[3px] font-mono text-[10px] text-[#8D9AAA] backdrop-blur-md">
          <span className="mr-1.5 uppercase text-[#667487]">SCALE:</span>
          {[5, 10, 20].map((s) => (
            <button
              key={s}
              onClick={() => onChangeVisualScale(s)}
              className={`px-1.5 py-0.5 rounded-[2px] cursor-pointer transition font-mono ${
                visualScale === s
                  ? "bg-[#28D7FF]/20 text-[#28D7FF] font-bold border border-[#28D7FF]/40"
                  : "text-[#8D9AAA] hover:text-[#E8EDF3]"
              }`}
            >
              {s}×
            </button>
          ))}
          <span className="text-[9px] text-[#667487] ml-1 hidden sm:inline">VISUAL</span>
        </div>
      </div>
    </div>
  );
};
