/**
 * Sketched primitives for the landing page.
 *
 * Rough.js draws the elements whose dimensions vary at runtime (card and
 * code-block borders, buttons that wrap variable-length labels). Everything
 * else — circled numbers, doodle icons, arrows, the wavy underline — is
 * hand-crafted SVG so the strokes look intentional rather than randomly
 * jittered.
 */
import {
  CSSProperties,
  ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import rough from "roughjs";

const INK = "#1A1A2E";
const RUST = "#C2532E";
const HIGHLIGHT = "#F5E6A8";
const PAPER = "#F7F4EC";

/* ─── Generic Rough overlay that re-draws on size change ───────────────── */

interface RoughDrawProps {
  draw: (rc: ReturnType<typeof rough.svg>, w: number, h: number) => SVGElement[];
  className?: string;
  style?: CSSProperties;
}

function RoughOverlay({ draw, className, style }: RoughDrawProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useLayoutEffect(() => {
    if (!svgRef.current) return;
    const el = svgRef.current.parentElement;
    if (!el) return;
    const update = () => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.round(r.width), h: Math.round(r.height) });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!svgRef.current || size.w === 0 || size.h === 0) return;
    const svg = svgRef.current;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const rc = rough.svg(svg);
    for (const node of draw(rc, size.w, size.h)) {
      svg.appendChild(node);
    }
  }, [draw, size]);

  return (
    <svg
      ref={svgRef}
      className={className}
      style={style}
      width={size.w || "100%"}
      height={size.h || "100%"}
      viewBox={`0 0 ${size.w || 1} ${size.h || 1}`}
      preserveAspectRatio="none"
      aria-hidden
    />
  );
}

/* ─── Card frame (index-card style border) ─────────────────────────────── */

export function SketchedCard({
  children,
  ink = INK,
  className,
}: {
  children: ReactNode;
  ink?: string;
  className?: string;
}) {
  const draw = (rc: ReturnType<typeof rough.svg>, w: number, h: number) => [
    rc.rectangle(4, 4, Math.max(10, w - 8), Math.max(10, h - 8), {
      stroke: ink,
      strokeWidth: 1.4,
      roughness: 1.6,
      bowing: 1.2,
      fill: "transparent",
    }),
  ];
  return (
    <div className={`lp-card ${className ?? ""}`}>
      <RoughOverlay className="lp-card-svg" draw={draw} />
      {children}
    </div>
  );
}

/* ─── Code block frame ─────────────────────────────────────────────────── */

export function SketchedCodeFrame({ children }: { children: ReactNode }) {
  // Stroke only — no fill. A solid fill would paint over the <pre> because
  // the overlay is position:absolute and the pre is in normal flow.
  const draw = (rc: ReturnType<typeof rough.svg>, w: number, h: number) => [
    rc.rectangle(3, 3, Math.max(10, w - 6), Math.max(10, h - 6), {
      stroke: INK,
      strokeWidth: 1.2,
      roughness: 1.8,
      bowing: 1.4,
      fill: "transparent",
    }),
  ];
  return (
    <div className="lp-code">
      <RoughOverlay className="lp-code-svg" draw={draw} />
      {children}
    </div>
  );
}

/* ─── Buttons (filled and outline) ─────────────────────────────────────── */

export function SketchedButton({
  children,
  variant = "primary",
  href,
  onClick,
}: {
  children: ReactNode;
  variant?: "primary" | "ghost";
  href?: string;
  onClick?: (e: React.MouseEvent) => void;
}) {
  // Both variants use a stroke-only Rough rectangle on top of CSS-supplied
  // background + border. Primary is heavier (thicker stroke). The CSS gives
  // each variant its fill colour so the button is always visible even if
  // the Rough overlay hasn't drawn yet.
  const draw = (rc: ReturnType<typeof rough.svg>, w: number, h: number) => [
    rc.rectangle(2, 2, Math.max(10, w - 4), Math.max(10, h - 4), {
      stroke: INK,
      strokeWidth: variant === "primary" ? 1.8 : 1.3,
      roughness: 1.4,
      bowing: 1,
      fill: "transparent",
    }),
  ];

  const inner = (
    <>
      <RoughOverlay className="lp-btn-svg" draw={draw} />
      <span>{children}</span>
    </>
  );

  if (href) {
    const external = href.startsWith("http");
    return (
      <a
        className={`lp-btn lp-btn-${variant}`}
        href={href}
        onClick={onClick}
        target={external ? "_blank" : undefined}
        rel={external ? "noreferrer" : undefined}
      >
        {inner}
      </a>
    );
  }
  return (
    <button className={`lp-btn lp-btn-${variant}`} onClick={onClick}>
      {inner}
    </button>
  );
}

/* ─── Hand-drawn brand mark ─────────────────────────────────────────────── */

export function BrandMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden>
      {/* Wobbly circle */}
      <path
        d="M14 3.4 C 21.5 3.6 25.1 8.8 24.6 14.5 C 24.0 20.4 19.4 24.7 13.6 24.5 C 7.8 24.3 3.4 19.6 3.6 13.7 C 3.8 8.0 7.6 3.6 14 3.4 Z"
        fill="none"
        stroke={INK}
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      {/* Knotted graph inside */}
      <circle cx="10" cy="11" r="1.3" fill={INK} />
      <circle cx="18" cy="11" r="1.3" fill={INK} />
      <circle cx="14" cy="18" r="1.3" fill={RUST} />
      <path
        d="M10 11 Q 14 9 18 11 M10 11 Q 11 16 14 18 M18 11 Q 17 16 14 18"
        fill="none"
        stroke={INK}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ─── Wavy underline (rust accent) ─────────────────────────────────────── */

export function WavyUnderline({
  color = RUST,
  strokeWidth = 2.6,
}: {
  color?: string;
  strokeWidth?: number;
}) {
  // Two slightly offset wobbly strokes to fake an ink-bleed double pass.
  return (
    <svg viewBox="0 0 200 14" preserveAspectRatio="none" aria-hidden>
      <path
        d="M2 8 Q 16 2 32 7 T 66 7 T 100 6 T 134 8 T 168 6 T 198 7"
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity="0.95"
      />
      <path
        d="M3 10 Q 18 5 34 9 T 68 9 T 102 8 T 136 10 T 170 8 T 197 10"
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth - 1.2}
        strokeLinecap="round"
        opacity="0.55"
      />
    </svg>
  );
}

/* ─── Wobbly horizontal rule ───────────────────────────────────────────── */

export function WobblyRule({
  color = INK,
  height = 8,
}: {
  color?: string;
  height?: number;
}) {
  return (
    <svg
      className="lp-nav-rule"
      viewBox="0 0 1200 8"
      preserveAspectRatio="none"
      aria-hidden
      style={{ height }}
    >
      <path
        d="M0 4 Q 60 2 120 4 T 240 5 T 360 3 T 480 4 T 600 5 T 720 3 T 840 4 T 960 5 T 1080 3 T 1200 4"
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.55"
      />
    </svg>
  );
}

/* ─── Hand-drawn checkmark ─────────────────────────────────────────────── */

export function Check({ color = RUST }: { color?: string }) {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden>
      <path
        d="M3 12 Q 6 14 9 16 Q 12 12 15 8 Q 17 6 19 4"
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ─── Circled number (1-6 hand-drawn ring with rust ink) ───────────────── */

export function CircledNumber({ n }: { n: number }) {
  return (
    <span className="lp-step-num" aria-hidden>
      <svg viewBox="0 0 56 56">
        {/* Two slightly offset rings for an ink-double-pass look */}
        <path
          d="M28 6 C 41 6 50 16 50 28 C 50 41 40 50 28 50 C 14 50 6 40 6 28 C 6 15 16 6 28 6 Z"
          fill="none"
          stroke={RUST}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M27 7 C 40 8 49 17 49 29 C 49 40 41 49 27 50 C 14 50 7 40 7 27 C 8 14 16 7 27 7 Z"
          fill="none"
          stroke={RUST}
          strokeWidth="0.9"
          strokeLinecap="round"
          opacity="0.45"
        />
      </svg>
      <span>{n}</span>
    </span>
  );
}

/* ─── Thumbtack doodle (top-right of cards) ────────────────────────────── */

export function Thumbtack() {
  return (
    <svg className="lp-card-thumbtack" viewBox="0 0 28 28" aria-hidden>
      {/* Pin head */}
      <ellipse cx="14" cy="9" rx="6.5" ry="5" fill={RUST} stroke={INK} strokeWidth="1.2" />
      <ellipse cx="12" cy="7.5" rx="2" ry="1.5" fill={PAPER} opacity="0.6" />
      {/* Pin shaft */}
      <path
        d="M14 14 L 14 24 M 12 22 L 14 26 L 16 22"
        fill="none"
        stroke={INK}
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ─── Clipboard "copy" icon ────────────────────────────────────────────── */

export function ClipboardIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden>
      <path
        d="M5 3 L 5 1.5 Q 5 1 5.5 1 L 10.5 1 Q 11 1 11 1.5 L 11 3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <path
        d="M3.5 3 Q 3 3 3 3.6 L 3 14 Q 3 14.6 3.6 14.6 L 12.4 14.6 Q 13 14.6 13 14 L 13 3.6 Q 13 3 12.4 3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ─── Doodle icon: tangled knowledge graph ─────────────────────────────── */

export function DoodleGraph() {
  return (
    <svg viewBox="0 0 96 96" width="96" height="96" aria-hidden>
      {/* Edges */}
      <g fill="none" stroke={INK} strokeWidth="1.4" strokeLinecap="round">
        <path d="M20 28 Q 36 16 56 22" />
        <path d="M20 28 Q 26 50 44 60" />
        <path d="M56 22 Q 70 36 72 56" />
        <path d="M44 60 Q 58 64 72 56" />
        <path d="M44 60 Q 30 72 22 80" />
        <path d="M72 56 Q 80 70 78 82" />
        <path d="M56 22 Q 60 40 44 60" opacity="0.6" />
      </g>
      {/* Nodes */}
      <circle cx="20" cy="28" r="4.4" fill={PAPER} stroke={INK} strokeWidth="1.4" />
      <circle cx="56" cy="22" r="4.4" fill={RUST} stroke={INK} strokeWidth="1.4" />
      <circle cx="44" cy="60" r="4.4" fill={PAPER} stroke={INK} strokeWidth="1.4" />
      <circle cx="72" cy="56" r="4.4" fill={PAPER} stroke={INK} strokeWidth="1.4" />
      <circle cx="22" cy="80" r="4.4" fill={PAPER} stroke={INK} strokeWidth="1.4" />
      <circle cx="78" cy="82" r="4.4" fill={PAPER} stroke={INK} strokeWidth="1.4" />
    </svg>
  );
}

/* ─── Doodle icon: branching tree with pruned branches ─────────────────── */

export function DoodleTree() {
  return (
    <svg viewBox="0 0 96 96" width="96" height="96" aria-hidden>
      {/* Trunk + branches */}
      <g fill="none" stroke={INK} strokeWidth="1.4" strokeLinecap="round">
        <path d="M48 86 L 48 56" />
        <path d="M48 56 Q 32 48 22 36" />
        <path d="M48 56 Q 64 48 74 36" />
        <path d="M22 36 Q 14 28 12 18" />
        <path d="M22 36 Q 28 26 32 18" />
        <path d="M74 36 Q 70 26 64 18" />
        <path d="M74 36 Q 80 26 84 18" />
      </g>
      {/* Leaf nodes — best-path one in rust */}
      <circle cx="12" cy="18" r="3.6" fill={PAPER} stroke={INK} strokeWidth="1.3" />
      <circle cx="32" cy="18" r="3.6" fill={PAPER} stroke={INK} strokeWidth="1.3" />
      <circle cx="64" cy="18" r="3.6" fill={RUST} stroke={INK} strokeWidth="1.3" />
      <circle cx="84" cy="18" r="3.6" fill={PAPER} stroke={INK} strokeWidth="1.3" />
      {/* Crossed-out branch to show pruning */}
      <g stroke={RUST} strokeWidth="1.3" strokeLinecap="round">
        <path d="M16 14 L 28 22" />
        <path d="M28 14 L 16 22" />
      </g>
    </svg>
  );
}

/* ─── Doodle icon: bar chart with arrow at the tallest bar ─────────────── */

export function DoodleChart() {
  return (
    <svg viewBox="0 0 96 96" width="96" height="96" aria-hidden>
      {/* Axes */}
      <path
        d="M16 14 L 16 80 L 88 80"
        fill="none"
        stroke={INK}
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      {/* Bars */}
      <rect x="24" y="58" width="10" height="22" fill="none" stroke={INK} strokeWidth="1.3" />
      <rect x="40" y="44" width="10" height="36" fill="none" stroke={INK} strokeWidth="1.3" />
      <rect x="56" y="32" width="10" height="48" fill={RUST} stroke={INK} strokeWidth="1.3" />
      <rect x="72" y="50" width="10" height="30" fill="none" stroke={INK} strokeWidth="1.3" />
      {/* Arrow pointing at the tallest bar */}
      <g stroke={RUST} strokeWidth="1.4" fill="none" strokeLinecap="round">
        <path d="M82 18 Q 70 18 64 28" />
        <path d="M62 26 L 64 30 L 68 28" />
      </g>
      {/* "best" label */}
      <text
        x="80"
        y="14"
        fill={RUST}
        fontSize="11"
        fontFamily="Caveat, cursive"
        fontStyle="italic"
        textAnchor="middle"
      >
        best
      </text>
    </svg>
  );
}

/* ─── Hero flow diagram: Documents → Agent → Winning Config ────────────── */

export function FlowDiagram() {
  return (
    <svg viewBox="0 0 360 240" width="100%" height="100%" aria-hidden style={{ maxWidth: 380 }}>
      {/* Three boxes */}
      <g fill="none" stroke={INK} strokeWidth="1.5" strokeLinecap="round">
        <path d="M20 90 Q 18 64 44 60 L 96 56 Q 122 56 122 82 L 122 138 Q 118 162 92 158 L 38 158 Q 16 156 20 134 Z" />
        <path d="M150 76 Q 148 50 174 50 L 220 48 Q 250 50 250 78 L 250 156 Q 246 178 222 174 L 168 172 Q 146 170 150 144 Z" />
        <path d="M280 90 Q 278 66 306 64 L 348 62 Q 354 84 354 110 L 354 154 Q 352 176 322 172 L 282 168 Q 270 164 280 142 Z" />
      </g>
      {/* Box content sketches */}
      {/* Documents — three lines */}
      <g stroke={INK} strokeWidth="1.2" strokeLinecap="round" fill="none">
        <path d="M40 92 L 100 92" />
        <path d="M40 106 L 96 108" />
        <path d="M40 122 L 100 120" />
        <path d="M40 138 L 80 138" />
      </g>
      {/* Agent — gear + arrows */}
      <g stroke={INK} strokeWidth="1.3" fill="none" strokeLinecap="round">
        <circle cx="200" cy="112" r="22" />
        <path d="M200 88 L 200 80 M200 144 L 200 152 M176 112 L 168 112 M232 112 L 224 112" />
        <path d="M183 95 L 178 90 M222 95 L 227 90 M183 129 L 178 134 M222 129 L 227 134" />
      </g>
      {/* Winning Config — checkmark */}
      <path
        d="M295 110 Q 305 122 312 130 Q 322 116 332 102 Q 338 96 342 90"
        fill="none"
        stroke={RUST}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {/* Arrows between boxes */}
      <g stroke={INK} strokeWidth="1.4" fill="none" strokeLinecap="round">
        <path d="M126 110 Q 138 108 146 110" />
        <path d="M142 105 L 148 110 L 142 115" />
        <path d="M254 112 Q 268 110 276 112" />
        <path d="M272 107 L 278 112 L 272 117" />
      </g>
      {/* Captions */}
      <g
        fill={INK}
        fontSize="12"
        fontFamily="Fraunces, serif"
        fontStyle="italic"
        textAnchor="middle"
      >
        <text x="71" y="190">documents</text>
        <text x="200" y="200">agent</text>
        <text x="316" y="190">best config</text>
      </g>
    </svg>
  );
}

/* ─── Connecting curves under feature grid ─────────────────────────────── */

export function FeatureFlowArrows() {
  return (
    <svg
      className="lp-features-flow"
      viewBox="0 0 1080 60"
      preserveAspectRatio="none"
      aria-hidden
      style={{ width: "100%", height: 60 }}
    >
      {/* Curve from card 1 → card 2 */}
      <g stroke={INK} strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.55">
        <path d="M180 8 Q 280 50 380 12" strokeDasharray="3 4" />
        <path d="M373 8 L 380 12 L 376 18" />
      </g>
      {/* Curve from card 2 → card 3 */}
      <g stroke={INK} strokeWidth="1.2" fill="none" strokeLinecap="round" opacity="0.55">
        <path d="M540 12 Q 660 50 760 8" strokeDasharray="3 4" />
        <path d="M753 4 L 760 8 L 756 14" />
      </g>
    </svg>
  );
}

/* ─── Final-section flourish (signoff doodle) ──────────────────────────── */

export function Flourish() {
  return (
    <svg className="lp-flourish" viewBox="0 0 80 36" aria-hidden>
      <path
        d="M4 20 Q 14 6 26 22 Q 36 32 48 20 Q 58 8 70 20 L 74 20"
        fill="none"
        stroke={RUST}
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path d="M68 16 L 74 20 L 68 24" fill="none" stroke={RUST} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/* ─── Footer flourish under name ───────────────────────────────────────── */

export function FooterLine() {
  return (
    <svg className="lp-footer-line" viewBox="0 0 220 8" preserveAspectRatio="none" aria-hidden>
      <path
        d="M2 4 Q 30 1 60 4 T 120 5 T 180 3 T 218 4"
        fill="none"
        stroke={RUST}
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* ─── GitHub mark (kept to one inline icon, ink stroke) ────────────────── */

export function GitHubMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
      />
    </svg>
  );
}

/* ─── Highlight wrapper (yellow marker behind text) ────────────────────── */

export function Highlight({ children }: { children: ReactNode }) {
  return (
    <span
      style={{
        background: HIGHLIGHT,
        padding: "0 4px",
        boxDecorationBreak: "clone",
        WebkitBoxDecorationBreak: "clone",
      }}
    >
      {children}
    </span>
  );
}
