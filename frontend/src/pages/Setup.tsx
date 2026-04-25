/**
 * Setup — public landing page (microsoft.ai-inspired).
 *
 * Used in two modes:
 *   - standalone (Vercel build, VITE_STATIC_LANDING=1) — full-bleed hero,
 *     no app sidebar, this is the public face of the project.
 *   - in-app (`/setup` route in dev) — same content rendered inside the
 *     Notion-style app shell.
 *
 * Lives in its own visual language (.lp-* classes in styles.css) on purpose:
 * the workbench inside the app stays Notion-flat; the landing carries the
 * hero, gradients, and motion that "first impression" pages call for.
 */
import { useState } from "react";

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
      "First install pulls roughly 2 GB (PyTorch, transformers, chromadb).",
  },
  {
    n: 3,
    title: "Add your Anthropic API key",
    body:
      "The agent uses Claude for test generation, RAGAS judging, and answer generation. Your key never leaves your local .env file.",
    code: `cp .env.example .env
# then edit .env and set:
ANTHROPIC_API_KEY=sk-ant-...`,
    hint:
      "No key yet? Create one at console.anthropic.com → API Keys. A few dollars of credit covers a full optimization run.",
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
      "In a second terminal — installs the React dependencies and launches Vite. The dev server proxies /api → :8000 automatically.",
    code: `cd frontend
npm install
npm run dev`,
  },
];

const FEATURES = [
  {
    icon: "🧬",
    title: "Synthetic test generation",
    body:
      "Knowledge graph extraction → seed Q&A → Evol-Instruct evolution → groundedness filter → 2D difficulty matrix. Build an evaluation set from your own documents, no hand labelling required.",
  },
  {
    icon: "⚡",
    title: "BFTS auto-optimization",
    body:
      "A four-stage tree-search agent (Preliminary → Baseline → Exploration → Ablation) explores the configuration space, prunes failing branches, and isolates the components that actually carried the win.",
  },
  {
    icon: "📊",
    title: "Results & export",
    body:
      "Live trajectory, RAGAS metric breakdown by difficulty and question type, ablation contributions, exportable LangChain or LlamaIndex snippet for the winning config.",
  },
];

interface SetupProps {
  standalone?: boolean;
}

export default function Setup({ standalone = false }: SetupProps) {
  return (
    <div className="lp">
      {standalone && <Nav />}
      <Hero />
      <FeaturesSection />
      <SetupSection />
      <LocalLLMSection />
      <CTABlock />
      {standalone && <Footer />}
    </div>
  );
}

function Nav() {
  return (
    <nav className="lp-nav">
      <div className="lp-brand">
        <span className="lp-brand-mark">◇</span>
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
  );
}

function Hero() {
  return (
    <header className="lp-hero">
      <div className="lp-hero-mesh" aria-hidden />
      <div className="lp-hero-inner">
        <span className="lp-eyebrow">
          <span className="lp-eyebrow-dot" />
          Open source · Local-first · Bring your own key
        </span>
        <h1 className="lp-title">
          Find the best RAG config{" "}
          <span className="lp-title-grad">automatically.</span>
        </h1>
        <p className="lp-sub">
          An agent that uploads your documents, synthesizes a test set, and
          searches the configuration space until it converges on the pipeline
          that scores highest on your data.
        </p>
        <div className="lp-cta-row">
          <a
            href="#setup"
            className="lp-btn lp-btn-primary"
            onClick={(e) => {
              e.preventDefault();
              document
                .getElementById("setup")
                ?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            Get started <Arrow />
          </a>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="lp-btn lp-btn-ghost"
          >
            <GitHubMark /> View on GitHub
          </a>
        </div>
        <div className="lp-hero-tags">
          <span className="lp-hero-tag">
            <span className="lp-hero-tag-dot" /> No hosted version
          </span>
          <span className="lp-hero-tag">
            <span className="lp-hero-tag-dot" /> Runs entirely on your machine
          </span>
          <span className="lp-hero-tag">
            <span className="lp-hero-tag-dot" /> Your API key, your data
          </span>
        </div>
      </div>
    </header>
  );
}

function FeaturesSection() {
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
      <div className="lp-feature-grid">
        {FEATURES.map((f) => (
          <div key={f.title} className="lp-feature">
            <div className="lp-feature-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function SetupSection() {
  return (
    <section className="lp-section" id="setup">
      <div className="lp-section-eyebrow">Get started</div>
      <h2 className="lp-section-title">Up and running in six steps.</h2>
      <p className="lp-section-sub">
        Clone, install, paste your key, start two servers. Everything runs on
        localhost — your documents and your key never leave your machine.
      </p>

      <div className="lp-prereqs">
        <Prereq label="Python" value="3.10 or newer" />
        <Prereq label="Node.js" value="18 or newer" />
        <Prereq label="Disk" value="~3 GB free" />
        <Prereq label="API key" value="Anthropic console" />
      </div>

      <div className="lp-callout">
        <strong>Bring your own API key.</strong> All LLM calls hit Anthropic
        directly from your machine. There is no proxy, no telemetry, no hosted
        backend — your key sits in a local <code>.env</code> file and never
        leaves it.
      </div>

      <div className="lp-steps">
        {STEPS.map((s) => (
          <StepCard key={s.n} step={s} />
        ))}
      </div>
    </section>
  );
}

function LocalLLMSection() {
  return (
    <section className="lp-section">
      <div className="lp-section-eyebrow">Optional</div>
      <h2 className="lp-section-title">Prefer to run a local LLM?</h2>
      <p className="lp-section-sub">
        Every generative component talks OpenAI-compatible HTTP, so any local
        server works. Skip step 3 above and add this to your <code>.env</code>{" "}
        instead.
      </p>
      <div style={{ maxWidth: 640 }}>
        <CodeBlock
          code={`# pull and serve the model
ollama pull qwen2.5:7b-instruct
ollama serve

# .env entries
LLM_PROVIDER=local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b-instruct`}
        />
      </div>
    </section>
  );
}

function CTABlock() {
  return (
    <section className="lp-cta-block">
      <h2>Ready to start?</h2>
      <p>
        Clone the repository, paste in your Anthropic key, run two servers, and
        the agent takes it from there.
      </p>
      <div
        className="lp-cta-row"
        style={{ justifyContent: "center", display: "flex" }}
      >
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="lp-btn lp-btn-primary"
        >
          <GitHubMark /> Clone the repository
        </a>
        <a
          href="#setup"
          className="lp-btn lp-btn-ghost"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("setup")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          Re-read the steps
        </a>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="lp-footer">
      <div>RAG Optimizer · MIT License · Stanislav Li</div>
      <div style={{ display: "flex", gap: 18 }}>
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          GitHub
        </a>
        <a href="#features">Features</a>
        <a href="#setup">Get started</a>
      </div>
    </footer>
  );
}

function Prereq({ label, value }: { label: string; value: string }) {
  return (
    <div className="lp-prereq">
      <div className="lp-prereq-label">{label}</div>
      <div className="lp-prereq-value">{value}</div>
    </div>
  );
}

function StepCard({ step }: { step: Step }) {
  return (
    <div className="lp-step">
      <div className="lp-step-head">
        <span className="lp-step-num">{step.n}</span>
        <h3 className="lp-step-title">{step.title}</h3>
      </div>
      <p className="lp-step-body">{step.body}</p>
      {step.code && <CodeBlock code={step.code} />}
      {step.hint && <p className="lp-step-hint">{step.hint}</p>}
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // Older browsers without clipboard API — silent no-op.
    }
  };
  return (
    <div className="lp-code">
      <pre>{code}</pre>
      <button
        onClick={onCopy}
        className="lp-code-copy"
        aria-label="Copy to clipboard"
      >
        {copied ? "copied" : "copy"}
      </button>
    </div>
  );
}

function Arrow() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path
        d="M2 7h10M8 3l4 4-4 4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function GitHubMark() {
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
