"use client";

import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Activity, ShieldCheck, Zap, ArrowRight, Play } from "lucide-react";
import { cn } from "@/lib/utils";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const INJECTED_SEISMIC_STYLES = `
  .gsap-reveal { visibility: hidden; }

  /* Film Grain Texture */
  .film-grain {
      position: absolute; inset: 0; width: 100%; height: 100%;
      pointer-events: none; z-index: 50; opacity: 0.04; mix-blend-mode: overlay;
      background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="noiseFilter"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(%23noiseFilter)"/></svg>');
  }

  /* Forest Grid Theme */
  .bg-seismic-grid {
      background-size: 48px 48px;
      background-image: 
          linear-gradient(to right, rgba(115, 230, 181, 0.06) 1px, transparent 1px),
          linear-gradient(to bottom, rgba(115, 230, 181, 0.06) 1px, transparent 1px);
      mask-image: radial-gradient(ellipse at center, black 0%, transparent 75%);
      -webkit-mask-image: radial-gradient(ellipse at center, black 0%, transparent 75%);
  }

  /* Physical Matte Materials */
  .text-mint-matte {
      color: #E8E8DE;
      text-shadow: 0 10px 30px rgba(115, 230, 181, 0.15), 0 2px 4px rgba(0, 0, 0, 0.6);
  }

  .text-emerald-silver {
      background: linear-gradient(180deg, #FFFFFF 0%, #73E6B5 50%, #2A7056 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      filter: drop-shadow(0px 10px 24px rgba(115, 230, 181, 0.2)) drop-shadow(0px 2px 4px rgba(0, 0, 0, 0.7));
  }

  /* Workstation Card Depth */
  .seismic-depth-card {
      background: linear-gradient(145deg, #0E1B17 0%, #07110F 100%);
      box-shadow: 
          0 40px 100px -20px rgba(0, 0, 0, 0.95),
          0 20px 40px -20px rgba(0, 0, 0, 0.8),
          inset 0 1px 2px rgba(115, 230, 181, 0.25),
          inset 0 -2px 4px rgba(0, 0, 0, 0.9);
      border: 1px solid rgba(115, 230, 181, 0.12);
      position: relative;
  }

  .card-sheen-seismic {
      position: absolute; inset: 0; border-radius: inherit; pointer-events: none; z-index: 50;
      background: radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(115, 230, 181, 0.08) 0%, transparent 45%);
      mix-blend-mode: screen; transition: opacity 0.3s ease;
  }

  .console-bezel {
      background-color: #07110F;
      box-shadow: 
          inset 0 0 0 2px rgba(115, 230, 181, 0.2), 
          inset 0 0 0 7px #050C0A, 
          0 40px 80px -15px rgba(0, 0, 0, 0.95),
          0 15px 25px -5px rgba(0, 0, 0, 0.8);
      transform-style: preserve-3d;
  }

  .floating-seismic-badge {
      background: linear-gradient(135deg, rgba(14, 27, 23, 0.85) 0%, rgba(7, 17, 15, 0.6) 100%);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 
          0 0 0 1px rgba(115, 230, 181, 0.2),
          0 20px 40px -10px rgba(0, 0, 0, 0.8),
          inset 0 1px 1px rgba(115, 230, 181, 0.25);
  }

  .btn-seismic-primary {
      background: #73E6B5;
      color: #07110F;
      box-shadow: 0 4px 14px rgba(115, 230, 181, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.8);
      transition: all 0.3s ease;
  }
  .btn-seismic-primary:hover {
      transform: translateY(-2px);
      background: #5cd4a2;
      box-shadow: 0 8px 20px rgba(115, 230, 181, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.8);
  }

  .btn-seismic-secondary {
      background: #0E1B17;
      color: #E8E8DE;
      border: 1px solid rgba(255, 255, 255, 0.12);
      transition: all 0.3s ease;
  }
  .btn-seismic-secondary:hover {
      transform: translateY(-2px);
      background: #152721;
      border-color: rgba(115, 230, 181, 0.3);
  }

  .progress-ring-seismic {
      transform: rotate(-90deg);
      transform-origin: center;
      stroke-dasharray: 377;
      stroke-dashoffset: 377;
      stroke-linecap: round;
  }
`;

export interface SeismicCinematicHeroProps extends React.HTMLAttributes<HTMLDivElement> {
  onExploreSimulator?: () => void;
  onExploreBenchmark?: () => void;
  onExploreLive?: () => void;
}

export function SeismicCinematicHero({
  onExploreSimulator,
  onExploreBenchmark,
  onExploreLive,
  className,
  ...props
}: SeismicCinematicHeroProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCardRef = useRef<HTMLDivElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef<number>(0);

  // High-performance dynamic mouse lighting
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (window.scrollY > window.innerHeight * 2) return;

      cancelAnimationFrame(requestRef.current);
      requestRef.current = requestAnimationFrame(() => {
        if (mainCardRef.current && consoleRef.current) {
          const rect = mainCardRef.current.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const mouseY = e.clientY - rect.top;

          mainCardRef.current.style.setProperty("--mouse-x", `${mouseX}px`);
          mainCardRef.current.style.setProperty("--mouse-y", `${mouseY}px`);

          const xVal = (e.clientX / window.innerWidth - 0.5) * 2;
          const yVal = (e.clientY / window.innerHeight - 0.5) * 2;

          gsap.to(consoleRef.current, {
            rotationY: xVal * 10,
            rotationX: -yVal * 10,
            ease: "power3.out",
            duration: 1.2,
          });
        }
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // GSAP Cinematic scroll and reveal timeline
  useEffect(() => {
    const isMobile = window.innerWidth < 768;

    const ctx = gsap.context(() => {
      gsap.set(".seismic-track", { autoAlpha: 0, y: 50, scale: 0.9, filter: "blur(15px)" });
      gsap.set(".seismic-hero-title", { autoAlpha: 1, clipPath: "inset(0 100% 0 0)" });
      gsap.set(".seismic-main-card", { y: window.innerHeight + 150, autoAlpha: 1 });
      gsap.set([".card-seismic-left", ".card-seismic-right", ".console-wrapper", ".floating-seismic-badge"], { autoAlpha: 0 });
      gsap.set(".seismic-cta-wrapper", { autoAlpha: 0, scale: 0.85, filter: "blur(20px)" });

      const introTl = gsap.timeline({ delay: 0.2 });
      introTl
        .to(".seismic-track", { duration: 1.6, autoAlpha: 1, y: 0, scale: 1, filter: "blur(0px)", ease: "expo.out" })
        .to(".seismic-hero-title", { duration: 1.3, clipPath: "inset(0 0% 0 0)", ease: "power4.inOut" }, "-=0.9");

      const scrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: containerRef.current,
          start: "top top",
          end: "+=6000",
          pin: true,
          scrub: 1,
          anticipatePin: 1,
        },
      });

      scrollTl
        .to([".seismic-hero-text", ".bg-seismic-grid"], { scale: 1.1, filter: "blur(15px)", opacity: 0.15, ease: "power2.inOut", duration: 2 }, 0)
        .to(".seismic-main-card", { y: 0, ease: "power3.inOut", duration: 2 }, 0)
        .to(".seismic-main-card", { width: "100%", height: "100%", borderRadius: "0px", ease: "power3.inOut", duration: 1.5 })
        .fromTo(
          ".console-wrapper",
          { y: 250, z: -400, rotationX: 40, autoAlpha: 0, scale: 0.7 },
          { y: 0, z: 0, rotationX: 0, autoAlpha: 1, scale: 1, ease: "expo.out", duration: 2.2 },
          "-=0.8"
        )
        .to(".progress-ring-seismic", { strokeDashoffset: 45, duration: 1.8, ease: "power3.inOut" }, "-=1.0")
        .fromTo(".floating-seismic-badge", { y: 80, autoAlpha: 0, scale: 0.8 }, { y: 0, autoAlpha: 1, scale: 1, stagger: 0.2, ease: "back.out(1.4)", duration: 1.4 }, "-=1.8")
        .fromTo(".card-seismic-left", { x: -40, autoAlpha: 0 }, { x: 0, autoAlpha: 1, ease: "power4.out", duration: 1.4 }, "-=1.4")
        .fromTo(".card-seismic-right", { x: 40, autoAlpha: 0 }, { x: 0, autoAlpha: 1, ease: "expo.out", duration: 1.4 }, "<")
        .to({}, { duration: 2.0 })
        .set(".seismic-hero-text", { autoAlpha: 0 })
        .set(".seismic-cta-wrapper", { autoAlpha: 1 })
        .to({}, { duration: 1.2 })
        .to([".console-wrapper", ".floating-seismic-badge", ".card-seismic-left", ".card-seismic-right"], {
          scale: 0.92, y: -30, autoAlpha: 0, ease: "power3.in", duration: 1.0, stagger: 0.04,
        })
        .to(
          ".seismic-main-card",
          {
            width: isMobile ? "94vw" : "88vw",
            height: isMobile ? "92vh" : "86vh",
            borderRadius: isMobile ? "24px" : "36px",
            ease: "expo.inOut",
            duration: 1.6,
          },
          "card-pull"
        )
        .to(".seismic-cta-wrapper", { scale: 1, filter: "blur(0px)", ease: "expo.inOut", duration: 1.6 }, "card-pull")
        .to(".seismic-main-card", { y: -window.innerHeight - 250, ease: "power3.in", duration: 1.4 });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn("relative w-screen h-screen overflow-hidden flex items-center justify-center bg-[#07110F] text-[#E8E8DE] font-sans antialiased select-none", className)}
      style={{ perspective: "1500px" }}
      {...props}
    >
      <style dangerouslySetInnerHTML={{ __html: INJECTED_SEISMIC_STYLES }} />
      <div className="film-grain" aria-hidden="true" />
      <div className="bg-seismic-grid absolute inset-0 z-0 pointer-events-none opacity-60" aria-hidden="true" />

      {/* Layer 1: Massive Editorial Title */}
      <div className="seismic-hero-text absolute z-10 flex flex-col items-center justify-center text-center w-screen px-4 will-change-transform transform-style-3d">
        <div className="seismic-track gsap-reveal flex items-center gap-2 mb-3 px-3 py-1 rounded-full bg-[#17483A] text-[#73E6B5] border border-[#73E6B5]/30 text-xs font-mono font-semibold uppercase tracking-widest">
          <span className="w-2 h-2 rounded-full bg-[#73E6B5] animate-pulse" />
          Neural Operator Structural Dynamics
        </div>
        <h1 className="seismic-track gsap-reveal text-mint-matte text-4xl sm:text-6xl md:text-7xl lg:text-[5.5rem] font-light tracking-tight mb-2">
          Sub-2ms nonlinear structural dynamics,
        </h1>
        <h1 className="seismic-hero-title gsap-reveal text-emerald-silver text-4xl sm:text-6xl md:text-7xl lg:text-[5.5rem] font-extrabold tracking-tight font-mono">
          Continuous Fourier Neural Operator.
        </h1>
      </div>

      {/* Layer 2: Tactile Direct Action CTAs */}
      <div className="seismic-cta-wrapper absolute z-10 flex flex-col items-center justify-center text-center w-screen px-4 gsap-reveal pointer-events-auto will-change-transform">
        <div className="text-xs font-mono text-[#73E6B5] uppercase tracking-widest mb-2 font-bold">
          Verified Against OpenSeesPy C-Runtime
        </div>
        <h2 className="text-3xl sm:text-5xl md:text-6xl font-bold mb-4 tracking-tight text-[#E8E8DE] font-mono">
          Enter The Seismic Research Desk
        </h2>
        <p className="text-[#82928B] text-sm sm:text-base md:text-lg mb-8 max-w-2xl mx-auto font-sans leading-relaxed">
          Simulate 3D building sway under major earthquakes in &lt; 2 milliseconds, verify numerical convergence against C++ Newmark-β integration, and screen real-time global events from USGS.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 font-mono text-xs">
          <button
            onClick={onExploreSimulator}
            className="btn-seismic-primary px-6 py-3.5 rounded-xl font-bold flex items-center justify-center gap-2 cursor-pointer shadow-lg"
          >
            <Play size={14} className="fill-[#07110F]" />
            <span>Launch Structural Simulator</span>
            <ArrowRight size={14} />
          </button>

          <button
            onClick={onExploreBenchmark}
            className="btn-seismic-secondary px-6 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 cursor-pointer"
          >
            <ShieldCheck size={14} className="text-[#73E6B5]" />
            <span>OpenSeesPy Benchmark</span>
          </button>

          <button
            onClick={onExploreLive}
            className="btn-seismic-secondary px-6 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 cursor-pointer"
          >
            <Activity size={14} className="text-[#73E6B5]" />
            <span>Live USGS Screening</span>
          </button>
        </div>
      </div>

      {/* Layer 3: Physical Workstation Card Centerpiece */}
      <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none" style={{ perspective: "1500px" }}>
        <div
          ref={mainCardRef}
          className="seismic-main-card seismic-depth-card relative overflow-hidden gsap-reveal flex items-center justify-center pointer-events-auto w-[94vw] md:w-[88vw] h-[92vh] md:h-[86vh] rounded-[24px] md:rounded-[36px]"
        >
          <div className="card-sheen-seismic" aria-hidden="true" />

          <div className="relative w-full h-full max-w-7xl mx-auto px-4 lg:px-12 flex flex-col justify-evenly lg:grid lg:grid-cols-3 items-center lg:gap-8 z-10 py-6 lg:py-0">
            {/* Right: Big Brand Typo */}
            <div className="card-seismic-right gsap-reveal order-1 lg:order-3 flex justify-center lg:justify-end z-20 w-full">
              <div className="text-center lg:text-right">
                <span className="text-xs font-mono text-[#73E6B5] uppercase tracking-widest font-bold block mb-1">
                  Surrogate Architecture
                </span>
                <h2 className="text-5xl md:text-7xl lg:text-[7.5rem] font-black uppercase tracking-tighter text-emerald-silver font-mono">
                  SEISMO<br className="hidden lg:block" />FNO
                </h2>
              </div>
            </div>

            {/* Center: Seismic Workstation Console Mockup */}
            <div className="console-wrapper order-2 lg:order-2 relative w-full h-[380px] lg:h-[580px] flex items-center justify-center z-10" style={{ perspective: "1000px" }}>
              <div className="relative w-full h-full flex items-center justify-center transform scale-[0.75] md:scale-90 lg:scale-100">
                {/* 3D Console Bezel */}
                <div
                  ref={consoleRef}
                  className="relative w-[300px] h-[540px] rounded-[2.5rem] console-bezel flex flex-col will-change-transform transform-style-3d border border-white/[0.08]"
                >
                  {/* Console Screen Interior */}
                  <div className="absolute inset-[6px] bg-[#07110F] rounded-[2.2rem] overflow-hidden p-5 flex flex-col justify-between text-[#E8E8DE]">
                    {/* Top Status Bar */}
                    <div className="flex justify-between items-center text-[10px] font-mono border-b border-white/[0.08] pb-2">
                      <span className="flex items-center gap-1.5 text-[#73E6B5] font-bold">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5] animate-ping" />
                        SUB-2MS INFERENCE
                      </span>
                      <span className="text-[#82928B]">MPS GPU</span>
                    </div>

                    {/* Circular Drift Response Meter */}
                    <div className="relative w-44 h-44 mx-auto flex items-center justify-center my-2">
                      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 140 140">
                        <circle cx="70" cy="70" r="55" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
                        <circle cx="70" cy="70" r="55" fill="none" stroke="#73E6B5" strokeWidth="10" className="progress-ring-seismic" />
                      </svg>
                      <div className="text-center z-10 flex flex-col items-center">
                        <span className="text-3xl font-extrabold font-mono text-[#E8E8DE] tracking-tight">2.71</span>
                        <span className="text-[9px] font-mono text-[#73E6B5] font-semibold uppercase">mm Roof Drift</span>
                        <span className="text-[8px] font-mono text-[#82928B] mt-0.5">IO (Immediate Occupancy)</span>
                      </div>
                    </div>

                    {/* Waveform Strip */}
                    <div className="p-3 bg-[#0B1714] rounded-xl border border-white/[0.06] space-y-1.5">
                      <div className="flex justify-between text-[10px] font-mono text-[#82928B]">
                        <span>DISPLACEMENT u(t)</span>
                        <span className="text-[#73E6B5] font-bold">10.82% L₂ Error</span>
                      </div>
                      <div className="h-10 w-full flex items-center">
                        <svg className="w-full h-full" viewBox="0 0 200 40">
                          <path
                            d="M 0 20 Q 25 5, 50 20 T 100 20 T 150 20 T 200 20"
                            fill="none"
                            stroke="#D6B56D"
                            strokeWidth="1.5"
                            strokeDasharray="3 2"
                          />
                          <path
                            d="M 0 20 Q 25 3, 50 20 T 100 18 T 150 22 T 200 20"
                            fill="none"
                            stroke="#73E6B5"
                            strokeWidth="1.8"
                          />
                        </svg>
                      </div>
                      <div className="flex justify-between text-[9px] font-mono text-[#82928B]">
                        <span className="text-[#D6B56D]">• OpenSees C-Runtime</span>
                        <span className="text-[#73E6B5]">• SeismoFNO</span>
                      </div>
                    </div>

                    {/* Bottom Indicator */}
                    <div className="text-center text-[10px] font-mono text-[#82928B] pt-1">
                      BLD-RC-03 · 3-Story Moment Frame
                    </div>
                  </div>
                </div>

                {/* Floating Glass Telemetry Badges */}
                <div className="floating-seismic-badge absolute flex top-4 lg:top-8 left-[-20px] lg:left-[-70px] rounded-xl p-3 items-center gap-3 z-30">
                  <div className="w-8 h-8 rounded-lg bg-[#17483A] flex items-center justify-center text-[#73E6B5]">
                    <Zap size={16} />
                  </div>
                  <div className="font-mono text-left">
                    <p className="text-[#E8E8DE] text-xs font-bold">1,060 sim/s</p>
                    <p className="text-[#82928B] text-[10px]">Surrogate Throughput</p>
                  </div>
                </div>

                <div className="floating-seismic-badge absolute flex bottom-8 lg:bottom-14 right-[-20px] lg:right-[-70px] rounded-xl p-3 items-center gap-3 z-30">
                  <div className="w-8 h-8 rounded-lg bg-[#17483A] flex items-center justify-center text-[#73E6B5]">
                    <ShieldCheck size={16} />
                  </div>
                  <div className="font-mono text-left">
                    <p className="text-[#E8E8DE] text-xs font-bold">305 Tests Passing</p>
                    <p className="text-[#82928B] text-[10px]">SHA-256 Audited</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Left: Scientific Accountability Text */}
            <div className="card-seismic-left gsap-reveal order-3 lg:order-1 flex flex-col justify-center text-center lg:text-left z-20 w-full px-4 lg:px-0">
              <span className="text-xs font-mono text-[#73E6B5] font-bold uppercase tracking-wider mb-1">
                Rigorous Ground Truth Validation
              </span>
              <h3 className="text-[#E8E8DE] text-xl md:text-2xl lg:text-3xl font-bold mb-3 tracking-tight font-mono">
                Physics-Grounded Operator Learning
              </h3>
              <p className="hidden md:block text-[#82928B] text-xs md:text-sm font-sans leading-relaxed max-w-sm lg:max-w-none">
                Replacing traditional step-by-step numerical time-history iterations with continuous frequency-domain Fourier neural operators. Evaluated on 2,160 physical simulations with modal FiLM conditioning.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SeismicCinematicHero;
