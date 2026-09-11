"use client";

import React, { useEffect, useRef } from "react";
import * as THREE_MODULE from "three";

declare global {
  interface Window {
    THREE?: any;
  }
}

export interface ShaderAnimationProps extends React.HTMLAttributes<HTMLDivElement> {
  speed?: number;
  lineDensity?: number;
}

export function ShaderAnimation({ className = "", speed = 0.06, lineDensity = 0.0008, ...props }: ShaderAnimationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    camera: any;
    scene: any;
    renderer: any;
    uniforms: any;
    animationId: number | null;
  }>({
    camera: null,
    scene: null,
    renderer: null,
    uniforms: null,
    animationId: null,
  });

  useEffect(() => {
    let scriptElement: HTMLScriptElement | null = null;
    let isCleanedUp = false;

    const init = (THREE: any) => {
      if (!containerRef.current || isCleanedUp) return;

      const container = containerRef.current;
      container.innerHTML = "";

      // Initialize camera
      const camera = new THREE.Camera();
      camera.position.z = 1;

      // Initialize scene
      const scene = new THREE.Scene();

      // Create geometry (compatible across Three.js versions)
      const GeometryClass = THREE.PlaneGeometry || THREE.PlaneBufferGeometry;
      const geometry = new GeometryClass(2, 2);

      // Define uniforms
      const uniforms = {
        time: { type: "f", value: 1.0 },
        resolution: { type: "v2", value: new THREE.Vector2() },
      };

      // Vertex shader
      const vertexShader = `
        void main() {
          gl_Position = vec4( position, 1.0 );
        }
      `;

      // Fragment shader
      const fragmentShader = `
        #define TWO_PI 6.2831853072
        #define PI 3.14159265359

        precision highp float;
        uniform vec2 resolution;
        uniform float time;
          
        float random (in float x) {
            return fract(sin(x)*1e4);
        }
        float random (vec2 st) {
            return fract(sin(dot(st.xy,
                                 vec2(12.9898,78.233)))*
                43758.5453123);
        }
        
        varying vec2 vUv;

        void main(void) {
          vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
          
          vec2 fMosaicScal = vec2(4.0, 2.0);
          vec2 vScreenSize = vec2(256,256);
          uv.x = floor(uv.x * vScreenSize.x / fMosaicScal.x) / (vScreenSize.x / fMosaicScal.x);
          uv.y = floor(uv.y * vScreenSize.y / fMosaicScal.y) / (vScreenSize.y / fMosaicScal.y);       
            
          float t = time*${speed.toFixed(4)}+random(uv.x)*0.4;
          float lineWidth = ${lineDensity.toFixed(6)};

          vec3 color = vec3(0.0);
          for(int j = 0; j < 3; j++){
            for(int i=0; i < 5; i++){
              color[j] += lineWidth*float(i*i) / abs(fract(t - 0.01*float(j)+float(i)*0.01)*1.0 - length(uv));        
            }
          }

          gl_FragColor = vec4(color[2],color[1],color[0],1.0);
        }
      `;

      // Create material
      const material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
      });

      // Create mesh and add to scene
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      // Initialize renderer
      const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      container.appendChild(renderer.domElement);

      // Store references
      sceneRef.current = {
        camera,
        scene,
        renderer,
        uniforms,
        animationId: null,
      };

      // Handle resize
      const onWindowResize = () => {
        if (!container || !renderer.domElement) return;
        const rect = container.getBoundingClientRect();
        const width = rect.width || window.innerWidth;
        const height = rect.height || window.innerHeight;
        renderer.setSize(width, height);
        uniforms.resolution.value.x = renderer.domElement.width;
        uniforms.resolution.value.y = renderer.domElement.height;
      };

      onWindowResize();
      window.addEventListener("resize", onWindowResize, false);

      // Animation loop
      const animate = () => {
        if (isCleanedUp) return;
        sceneRef.current.animationId = requestAnimationFrame(animate);
        uniforms.time.value += 0.05;
        renderer.render(scene, camera);
      };

      animate();
    };

    // If local Three.js module is available or window.THREE exists, use it immediately
    if (typeof window !== "undefined" && window.THREE) {
      init(window.THREE);
    } else if (THREE_MODULE && THREE_MODULE.Scene) {
      init(THREE_MODULE);
    } else {
      // Fallback: load Three.js dynamically via CDN
      scriptElement = document.createElement("script");
      scriptElement.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/89/three.min.js";
      scriptElement.async = true;
      scriptElement.onload = () => {
        if (window.THREE) {
          init(window.THREE);
        }
      };
      document.head.appendChild(scriptElement);
    }

    return () => {
      isCleanedUp = true;
      if (sceneRef.current.animationId) {
        cancelAnimationFrame(sceneRef.current.animationId);
      }
      if (sceneRef.current.renderer) {
        try {
          sceneRef.current.renderer.dispose();
        } catch {
          // ignore cleanup errors
        }
      }
      if (scriptElement && scriptElement.parentNode) {
        try {
          scriptElement.parentNode.removeChild(scriptElement);
        } catch {
          // ignore
        }
      }
    };
  }, [speed, lineDensity]);

  return (
    <div
      ref={containerRef}
      className={`w-full h-full absolute inset-0 pointer-events-none ${className}`}
      {...props}
    />
  );
}

export default ShaderAnimation;
