import { ShaderAnimation } from "@/components/ui/shader-lines";

export default function ShaderLinesDemo() {
  return (
    <div className="relative flex h-[650px] w-full flex-col items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-[#07110F]">
      <ShaderAnimation />
      <span className="pointer-events-none z-10 text-center text-7xl leading-none font-semibold tracking-tighter whitespace-pre-wrap text-white font-mono drop-shadow-2xl">
        Shader Lines
      </span>
      <p className="pointer-events-none z-10 text-center text-sm text-[#73E6B5] font-mono mt-4 tracking-widest uppercase">
        Continuous Frequency Field & Wavefront Simulation
      </p>
    </div>
  );
}
