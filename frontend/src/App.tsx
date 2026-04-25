/**
 * App shell: Notion-style two-column layout.
 *
 *   ┌──────────┬─────────────────────────────────────────┐
 *   │ Sidebar  │  Page content                           │
 *   │ (fixed)  │  (router outlet)                        │
 *   └──────────┴─────────────────────────────────────────┘
 *
 * The sidebar mimics Notion's workspace sidebar: a faint background, a
 * workspace title at the top, and a flat list of nav entries. Active entry
 * gets the familiar `bg-hover` highlight. No collapsible groups yet — we
 * can add sub-pages later under "Results" if needed.
 */
import { CSSProperties } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { RunProvider, useRun } from "./context/RunContext";
import { TestGenProvider } from "./context/TestGenContext";
import AutoOptimize from "./pages/AutoOptimize";
import Comparison from "./pages/Comparison";
import ConfigLab from "./pages/ConfigLab";
import Export from "./pages/Export";
import Results from "./pages/Results";
import Setup from "./pages/Setup";
import Trajectory from "./pages/Trajectory";
import Upload from "./pages/Upload";
import { colors, font, radius, space } from "./theme";

interface NavItem {
  to: string;
  label: string;
  hint: string;
  icon: string;
}

const PRIMARY_NAV: NavItem[] = [
  { to: "/setup", label: "Get started", hint: "0", icon: "▶" },
  { to: "/upload", label: "Upload & TestGen", hint: "1", icon: "📄" },
  { to: "/optimize", label: "Auto-Optimize", hint: "2", icon: "⚡" },
  { to: "/results", label: "Results", hint: "3", icon: "📊" },
  { to: "/export", label: "Export", hint: "4", icon: "↓" },
];

const ADVANCED_NAV: NavItem[] = [
  { to: "/trajectory", label: "Trajectory", hint: "", icon: "↗" },
  { to: "/comparison", label: "Comparison", hint: "", icon: "⇄" },
  { to: "/config-lab", label: "Config Lab", hint: "", icon: "⚙" },
];

// VITE_STATIC_LANDING=1 strips the full app and renders only the Setup
// landing page. We use it for the public Vercel build, where there is no
// Django backend — the rest of the routes would 404 on every API call.
const STATIC_LANDING = import.meta.env.VITE_STATIC_LANDING === "1";

export default function App() {
  if (STATIC_LANDING) {
    return (
      <Routes>
        <Route path="*" element={<Setup standalone />} />
      </Routes>
    );
  }
  return (
    <RunProvider>
      <TestGenProvider>
        <div style={shell}>
          <Sidebar />
          <main style={main}>
            <Routes>
              <Route path="/" element={<Navigate to="/setup" replace />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/optimize" element={<AutoOptimize />} />
              <Route path="/results" element={<Results />} />
              <Route path="/export" element={<Export />} />
              <Route path="/trajectory" element={<Trajectory />} />
              <Route path="/comparison" element={<Comparison />} />
              <Route path="/config-lab" element={<ConfigLab />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
        </div>
      </TestGenProvider>
    </RunProvider>
  );
}

function Sidebar() {
  const { current } = useRun();
  return (
    <aside style={sidebar}>
      <div style={workspaceHeader}>
        <div style={workspaceIcon}>◇</div>
        <div>
          <div style={workspaceName}>RAG Optimizer</div>
          <div style={workspaceSub}>Experiment workspace</div>
        </div>
      </div>

      <NavGroup label="Pipeline" items={PRIMARY_NAV} />
      <NavGroup label="Advanced" items={ADVANCED_NAV} />

      <div style={{ flex: 1 }} />

      <div style={footerSection}>
        <div style={footerLabel}>Current run</div>
        {current ? (
          <div style={footerRunLine}>
            <span style={{ color: colors.text, fontWeight: 500 }}>
              {current.label}
            </span>
            <span style={{ color: colors.textFaint, fontSize: 11 }}>
              {current.bestScore !== null
                ? `score ${current.bestScore.toFixed(3)}`
                : "—"}
            </span>
          </div>
        ) : (
          <div style={{ color: colors.textFaint, fontSize: 12 }}>
            No runs yet
          </div>
        )}
      </div>
    </aside>
  );
}

function NavGroup({ label, items }: { label: string; items: NavItem[] }) {
  return (
    <div style={{ marginTop: space.lg }}>
      <div style={navGroupLabel}>{label}</div>
      {items.map((item) => (
        <NavEntry key={item.to} item={item} />
      ))}
    </div>
  );
}

function NavEntry({ item }: { item: NavItem }) {
  const loc = useLocation();
  const active = loc.pathname === item.to;
  return (
    <NavLink to={item.to} style={() => navEntryStyle(active)}>
      <span style={navIcon}>{item.icon}</span>
      <span style={{ flex: 1 }}>{item.label}</span>
      {item.hint && <span style={navHint}>{item.hint}</span>}
    </NavLink>
  );
}

function NotFound() {
  return (
    <div style={{ padding: space.xxl, color: colors.textMuted }}>
      <h2 style={{ fontWeight: 600 }}>Page not found</h2>
      <p>The path you requested doesn't exist in this workspace.</p>
    </div>
  );
}

const shell: CSSProperties = {
  display: "flex",
  minHeight: "100vh",
  background: colors.bg,
  fontFamily: font.sans,
};

const sidebar: CSSProperties = {
  width: 248,
  flexShrink: 0,
  padding: `${space.lg}px ${space.sm}px ${space.sm}px`,
  background: colors.bgSunken,
  borderRight: `1px solid ${colors.border}`,
  display: "flex",
  flexDirection: "column",
  position: "sticky",
  top: 0,
  maxHeight: "100vh",
  overflowY: "auto",
};

const main: CSSProperties = {
  flex: 1,
  minWidth: 0,
  background: colors.bg,
};

const workspaceHeader: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  padding: `${space.sm}px ${space.sm}px`,
  borderRadius: radius.md,
};

const workspaceIcon: CSSProperties = {
  width: 28,
  height: 28,
  display: "grid",
  placeItems: "center",
  borderRadius: radius.md,
  background: colors.accent,
  color: "#fff",
  fontSize: 16,
  fontWeight: 700,
};

const workspaceName: CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: colors.text,
  lineHeight: 1.2,
};

const workspaceSub: CSSProperties = {
  fontSize: 11,
  color: colors.textMuted,
  lineHeight: 1.2,
};

const navGroupLabel: CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textFaint,
  padding: `${space.xs}px ${space.sm}px`,
  marginBottom: 2,
};

const navEntryStyle = (active: boolean): CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  padding: `${space.xs + 2}px ${space.sm}px`,
  borderRadius: radius.md,
  color: active ? colors.text : colors.textMuted,
  background: active ? colors.bgHover : "transparent",
  fontSize: 14,
  fontWeight: active ? 500 : 400,
  textDecoration: "none",
  cursor: "pointer",
  transition: "background 80ms ease",
});

const navIcon: CSSProperties = {
  width: 18,
  textAlign: "center",
  fontSize: 13,
  color: colors.textMuted,
};

const navHint: CSSProperties = {
  fontFamily: font.mono,
  fontSize: 11,
  color: colors.textFaint,
  background: "transparent",
};

const footerSection: CSSProperties = {
  padding: `${space.sm}px`,
  borderTop: `1px solid ${colors.border}`,
  marginTop: space.lg,
};

const footerLabel: CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  color: colors.textFaint,
  marginBottom: space.xs,
};

const footerRunLine: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 2,
  fontSize: 13,
};
