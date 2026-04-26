/**
 * Setup — public landing page (hand-drawn editorial / engineer's notebook).
 *
 * Used in two modes:
 *   - standalone (Vercel build, VITE_STATIC_LANDING=1) — full-bleed page,
 *     no app sidebar, this is the public face of the project.
 *   - in-app (`/setup` route in dev) — same content rendered inside the
 *     Notion-style app shell.
 *
 * The visual language is its own (.lp-* in styles.css plus sketched
 * primitives in components/landing/sketch.tsx). The workbench inside the
 * app stays Notion-flat; the landing carries the warm-paper aesthetic.
 */
import { useState } from "react";

import {
  BrandMark,
  Check,
  CircledNumber,
  ClipboardIcon,
  DoodleChart,
  DoodleGraph,
  DoodleTree,
  FeatureFlowArrows,
  FlowDiagram,
  Flourish,
  FooterLine,
  GitHubMark,
  HeroToFeaturesArrow,
  Highlight,
  SketchedButton,
  SketchedCard,
  SketchedCodeFrame,
  WavyUnderline,
  WobblyRule,
} from "../components/landing/sketch";

const REPO_URL = "https://github.com/stanislavli3/rag-optimization-agent";

interface Step {
  n: number;
  title: string;
  body: string;
  code?: string;
  hint?: string;
}

const STEPS: Step[] = [
  {
    n: 1,
    title: "Clone the repository",
    body: "Pull the project onto your machine and step into it.",
    code: `git clone ${REPO_URL}.git
cd rag-optimization-agent`,
  },
  {
    n: 2,
    title: "Install Python dependencies",
    body:
      "Python 3.10+ is required. A virtual environment keeps the ML libraries isolated.",
    code: `python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm`,
    hint:
      "first install pulls roughly 2 GB — pytorch, transformers, chromadb",
  },
  {
    n: 3,
    title: "Add your Anthropic API key",
    body:
      "The agent uses Claude for test generation, RAGAS judging, and answer generation. Your key sits in a local .env file and never leaves it.",
    code: `cp .env.example .env
# then edit .env and set:
ANTHROPIC_API_KEY=sk-ant-...`,
    hint:
      "no key yet? make one at console.anthropic.com — a few dollars covers a full run",
  },
  {
    n: 4,
    title: "Run database migrations",
    body: "Sets up the SQLite tables Django uses to track experiments and stream events.",
    code: `python backend/manage.py migrate`,
  },
  {
    n: 5,
    title: "Start the backend",
    body: "Serves the JSON API and the live SSE event stream on port 8000.",
    code: `python backend/manage.py runserver`,
  },
  {
    n: 6,
    title: "Start the frontend",
    body:
      "In a second terminal — installs the React dependencies and launches Vite, which proxies /api → :8000 automatically.",
    code: `cd frontend
npm install
npm run dev`,
    hint: "open http://localhost:5173 — the agent is yours.",
  },
];

const FEATURES = [
  {
    Icon: DoodleGraph,
    title: "Synthetic test generation",
    body:
      "Knowledge graph extraction, seed Q&A, Evol-Instruct evolution, groundedness filter, 2D difficulty matrix — an evaluation set built from your own documents, no hand-labelling required.",
    caption: "fig. 1",
  },
  {
    Icon: DoodleTree,
    title: "BFTS auto-optimization",
    body:
      "A four-stage tree-search agent (Preliminary → Baseline → Exploration → Ablation) explores the configuration space, prunes failing branches, and isolates the components that carried the win.",
    caption: "fig. 2",
  },
  {
    Icon: DoodleChart,
    title: "Results & export",
    body:
      "Live trajectory, RAGAS metrics broken down by difficulty and question type, ablation contributions, exportable LangChain or LlamaIndex snippet for the winning config.",
    caption: "fig. 3",
  },
];

interface SetupProps {
  standalone?: boolean;
}

export default function Setup({ standalone = false }: SetupProps) {
  return (
    <div className="lp">
      <div className="lp-shell">
        {standalone && <Header />}
        <Hero />
        <div className="lp-hero-bridge-wrap">
          <HeroToFeaturesArrow />
        </div>
        <Features />
        <GetStarted />
        <LocalLLM />
        <FinalCTA />
        {standalone && <Footer />}
      </div>
    </div>
  );
}

/* ─── Header ────────────────────────────────────────────────────────── */
function Header() {
  return (
    <>
      <nav className="lp-nav">
        <div className="lp-brand">
          <BrandMark />
          <span>RAG Optimizer</span>
        </div>
        <div className="lp-nav-links">
          <a href="#features">Features</a>
          <a href="#setup">Get started</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub →
          </a>
        </div>
      </nav>
      <WobblyRule />
      <div className="lp-eyebrow">
        <span>Open source</span>
        <span className="lp-eyebrow-dot" />
        <span>Local-first</span>
        <span className="lp-eyebrow-dot" />
        <span>Bring your own key</span>
      </div>
    </>
  );
}

/* ─── Hero ──────────────────────────────────────────────────────────── */
function Hero() {
  const scrollToSetup = (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById("setup")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="lp-hero">
      <div className="lp-hero-copy">
        <h1 className="lp-title">
          Find the best RAG config —{" "}
          <span className="lp-title-mark">
            automatically
            <WavyUnderline />
          </span>
          .
        </h1>
        <p className="lp-sub">
          An agent that uploads your documents, synthesizes a test set, and
          searches the configuration space until it converges on the pipeline
          that scores highest on your data.
        </p>
        <div className="lp-cta-row">
          <SketchedButton variant="primary" href="#setup" onClick={scrollToSetup}>
            Get started
          </SketchedButton>
          <SketchedButton variant="ghost" href={REPO_URL}>
            <GitHubMark /> View on GitHub
          </SketchedButton>
        </div>
        <div className="lp-hero-tags">
          <span className="lp-hero-tag">
            <Check /> No hosted version
          </span>
          <span className="lp-hero-tag">
            <Check /> Runs entirely on your machine
          </span>
          <span className="lp-hero-tag">
            <Check /> Your API key, your data
          </span>
        </div>
      </div>
      <div className="lp-hero-diagram">
        <FlowDiagram />
      </div>
    </section>
  );
}

/* ─── Features ──────────────────────────────────────────────────────── */
function Features() {
  return (
    <section className="lp-section" id="features">
      <div className="lp-section-eyebrow">What it does</div>
      <h2 className="lp-section-title">
        From raw documents to a tuned RAG pipeline.
      </h2>
      <p className="lp-section-sub">
        Three modules wired into one workflow — generate the test set, let the
        agent optimise, ship the winning config.
      </p>
      <FeatureFlowArrows />
      <div className="lp-features">
        {FEATURES.map((f) => (
          <article key={f.title} className="lp-feature">
            <div className="lp-feature-icon">
              <f.Icon />
            </div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
            <p className="lp-feature-caption">{f.caption}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

/* ─── Get started ───────────────────────────────────────────────────── */
function GetStarted() {
  return (
    <section className="lp-section" id="setup">
      <div className="lp-section-eyebrow">Get started</div>
      <h2 className="lp-section-title">Up and running in six steps.</h2>
      <p className="lp-section-sub">
        Clone, install, paste your key, start two servers. Everything runs on{" "}
        <Highlight>localhost</Highlight> — your documents and your key never
        leave your machine.
      </p>

      <SketchedCard className="lp-callout-card">
        <p className="lp-card-title">Requirements</p>
        <div className="lp-prereqs">
          <Prereq label="Python" value="3.10 or newer" />
          <Prereq label="Node.js" value="18 or newer" />
          <Prereq label="Disk" value="~3 GB free" />
          <Prereq label="API key" value="Anthropic console" />
        </div>
      </SketchedCard>

      <div className="lp-steps">
        <div className="lp-steps-thread" aria-hidden />
        {STEPS.map((s) => (
          <StepRow key={s.n} step={s} />
        ))}
      </div>
    </section>
  );
}

function Prereq({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="lp-prereq-label">{label}</div>
      <div className="lp-prereq-value">{value}</div>
    </div>
  );
}

function StepRow({ step }: { step: Step }) {
  return (
    <div className="lp-step">
      <CircledNumber n={step.n} />
      <h3 className="lp-step-title">{step.title}</h3>
      <p className="lp-step-body">{step.body}</p>
      {step.code && (
        <SketchedCodeFrame>
          <pre>{step.code}</pre>
          <CopyButton code={step.code} />
        </SketchedCodeFrame>
      )}
      {step.hint && <p className="lp-step-hint">— {step.hint}</p>}
    </div>
  );
}

function CopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // No clipboard API — silent no-op.
    }
  };
  return (
    <button
      className="lp-code-copy"
      onClick={onCopy}
      aria-label="Copy code to clipboard"
    >
      <ClipboardIcon />
      {copied ? "copied" : "copy"}
    </button>
  );
}

/* ─── Optional local-LLM ────────────────────────────────────────────── */
function LocalLLM() {
  return (
    <section className="lp-section">
      <SketchedCard>
        <p className="lp-card-title">
          Optional — prefer to run a local LLM?
        </p>
        <p
          style={{
            margin: "0 0 16px",
            fontFamily: "Fraunces, serif",
            fontSize: 16,
            lineHeight: 1.6,
            color: "var(--lp-ink-soft)",
          }}
        >
          Every generative component talks OpenAI-compatible HTTP, so any
          local server works. Skip step&nbsp;3 above and add this to your{" "}
          <code style={{ background: "var(--lp-highlight)", padding: "1px 5px", borderRadius: 2, fontFamily: "JetBrains Mono, monospace", fontSize: 13.5 }}>.env</code>{" "}
          instead.
        </p>
        <SketchedCodeFrame>
          <pre>{`# pull and serve the model
ollama pull qwen2.5:7b-instruct
ollama serve

# .env entries
LLM_PROVIDER=local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b-instruct`}</pre>
          <CopyButton
            code={`ollama pull qwen2.5:7b-instruct
ollama serve

LLM_PROVIDER=local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b-instruct`}
          />
        </SketchedCodeFrame>
        <p
          style={{
            marginTop: 14,
            fontFamily: "Caveat, cursive",
            fontSize: 18,
            color: "var(--lp-rust)",
            fontStyle: "italic",
          }}
        >
          ↑ swap in any OpenAI-compatible endpoint
        </p>
      </SketchedCard>
    </section>
  );
}

/* ─── Final CTA ─────────────────────────────────────────────────────── */
function FinalCTA() {
  return (
    <section className="lp-final">
      <h2>Ready to start?</h2>
      <p>
        Clone the repository, paste in your Anthropic key, run two servers,
        and the agent takes it from there.
      </p>
      <div className="lp-final-row">
        <SketchedButton variant="primary" href={REPO_URL}>
          <GitHubMark /> Clone the repository
        </SketchedButton>
        <SketchedButton
          variant="ghost"
          href="#setup"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("setup")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          Re-read the steps
        </SketchedButton>
      </div>
      <Flourish />
    </section>
  );
}

/* ─── Footer ────────────────────────────────────────────────────────── */
function Footer() {
  return (
    <footer className="lp-footer">
      <FooterLine />
      <div>
        <span style={{ fontFamily: "Fraunces, serif", fontSize: 15, color: "var(--lp-ink)" }}>
          RAG Optimizer
        </span>{" "}
        · MIT License · Stanislav Li
      </div>
      <div className="lp-footer-links">
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          GitHub
        </a>
        <a href="#features">Features</a>
        <a href="#setup">Get started</a>
      </div>
    </footer>
  );
}
