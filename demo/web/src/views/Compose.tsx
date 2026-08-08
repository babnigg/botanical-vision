import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import NotYet from "../components/NotYet";
import { useStore } from "../store";
import type { ComposeResponse, ModuleState, PaletteEntry } from "../types";

const SUN_NAMES = ["shade", "part sun", "full sun"];
const LAYERS = ["structural", "seasonal", "filler", "groundcover"];
const COUNT_CYCLE: Record<string, number[]> = {
  structural: [0, 1, 2, 3],
  default: [0, 1, 3, 5, 7],
};

export default function Compose() {
  const { registry, setView, toolbox, addToToolbox } = useStore();
  const [palette, setPalette] = useState<PaletteEntry[]>([]);
  const [status, setStatus] = useState<ModuleState | null>(null);
  const [width, setWidth] = useState(5.5);
  const [depth, setDepth] = useState(2.6);
  const [sun, setSun] = useState(2);
  const [pins, setPins] = useState<Record<string, number>>({});
  const [result, setResult] = useState<ComposeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [renderState, setRenderState] = useState<string | null>(null);
  const [renderImg, setRenderImg] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
  }, []);

  const startRender = async () => {
    if (!result || !result.bed) return;
    setRenderState("starting");
    setRenderImg(null);
    try {
      const r = await api.composeRender({
        width: result.bed.w,
        depth: result.bed.d,
        plants: result.plants.map((p) => ({ species: p.species, x: p.x, y: p.y, r: p.r })),
      });
      if (!r.ok || !r.job) {
        setRenderState(`failed: ${r.error ?? "no job"}`);
        return;
      }
      const job = r.job;
      pollRef.current = window.setInterval(async () => {
        const st = await api.composeRenderStatus(job).catch(() => null);
        if (!st) return;
        if (st.status === "done" && st.png_b64) {
          setRenderImg(st.png_b64);
          setRenderState(null);
          if (pollRef.current) window.clearInterval(pollRef.current);
        } else if (st.status === "failed") {
          setRenderState(`failed: ${st.error ?? "unknown"}`);
          if (pollRef.current) window.clearInterval(pollRef.current);
        } else {
          setRenderState(`${st.status}… ${Math.round((st.elapsed ?? 0) / 60)} min elapsed`);
        }
      }, 10000);
    } catch (e) {
      setRenderState(e instanceof Error ? e.message : "render failed");
    }
  };

  useEffect(() => {
    api
      .composePalette()
      .then((r) => {
        setPalette(r.palette);
        setStatus(r.status);
      })
      .catch(() => setStatus(registry?.modules.compose ?? "prototype"));
  }, [registry]);

  if (status === "prototype")
    return (
      <NotYet
        title="Garden design"
        needs="the garden layout model (notebook 11) and its plant palette"
        onRoadmap={() => setView("overview")}
      />
    );

  const inToolbox = new Set(toolbox.map((e) => e.tax.species));

  const cyclePin = (p: PaletteEntry) => {
    const cycle = COUNT_CYCLE[p.layer] ?? COUNT_CYCLE.default;
    const cur = pins[p.name] ?? 0;
    const at = cycle.indexOf(cur);
    const next = cycle[(at + 1) % cycle.length];
    setPins((prev) => {
      const out = { ...prev };
      if (next === 0) delete out[p.name];
      else out[p.name] = next;
      return out;
    });
  };

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.compose({
        width,
        depth,
        sun,
        pins: Object.entries(pins).map(([species, count]) => ({ species, count })),
      });
      setResult(r);
      // pinned toolbox species are now planted
      toolbox.forEach((e) => {
        if (pins[e.tax.species]) addToToolbox(e.tax, { planted: true });
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "compose failed");
    } finally {
      setBusy(false);
    }
  };

  const bed = result?.bed;
  const pad = 0.25;

  return (
    <>
      <div className="hero">
        <div className="kick">Compose</div>
        <h1>Garden studio</h1>
        <p>
          Pick a bed, pin favorites, and let the layout model plant the rest — a
          masked-diffusion transformer over designed planting plans, sampled
          best-of-6 with constraint repair (notebook 13).
        </p>
      </div>
      <div className="grid2">
        <div className="card">
          <h4>
            Site <span className="aux">{SUN_NAMES[sun]}</span>
          </h4>
          <div className="ctlrow">
            <label>
              width <span className="num">{width.toFixed(1)} m</span>
              <input
                type="range"
                min={3.5}
                max={7.5}
                step={0.1}
                value={width}
                onChange={(e) => setWidth(Number(e.target.value))}
              />
            </label>
            <label>
              depth <span className="num">{depth.toFixed(1)} m</span>
              <input
                type="range"
                min={1.8}
                max={3.0}
                step={0.1}
                value={depth}
                onChange={(e) => setDepth(Number(e.target.value))}
              />
            </label>
            <label>
              sun
              <select value={sun} onChange={(e) => setSun(Number(e.target.value))}>
                {SUN_NAMES.map((s, i) => (
                  <option key={s} value={i}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <h4 style={{ marginTop: 18 }}>
            Pin plants <span className="aux">click to cycle the count</span>
          </h4>
          {LAYERS.map((layer) => (
            <div key={layer} className="pinrow">
              <span className="pinlayer">{layer}</span>
              <div className="pinchips">
                {palette
                  .filter((p) => p.layer === layer)
                  .map((p) => (
                    <button
                      key={p.name}
                      className={`pinchip ${pins[p.name] ? "on" : ""}`}
                      onClick={() => cyclePin(p)}
                      title={`${p.h} cm · ${SUN_NAMES[p.sun]}`}
                    >
                      <i style={{ background: p.color }} />
                      <span className="binom">{p.name}</span>
                      {pins[p.name] ? <b>×{pins[p.name]}</b> : null}
                      {inToolbox.has(p.name) ? <em title="in your toolbox">★</em> : null}
                    </button>
                  ))}
              </div>
            </div>
          ))}
          <div className="actions">
            <button className="btn" onClick={generate} disabled={busy}>
              {busy ? "Planting…" : result ? "Regenerate" : "Generate plan"}
            </button>
          </div>
          {error && <div className="note warn">{error}</div>}
        </div>
        <div className="card">
          <h4>
            Plan{" "}
            {result && (
              <span className="aux">
                served by the{" "}
                {result.served.startsWith("diffusion") ? result.served : "rule engine"}
              </span>
            )}
          </h4>
          {!result || !bed ? (
            <div className="empty">No plan yet — set the site and generate.</div>
          ) : (
            <>
              <div className="plan-wrap">
                <svg
                  viewBox={`${-pad} ${-pad} ${bed.w + 2 * pad} ${bed.d + 2 * pad}`}
                  preserveAspectRatio="xMidYMid meet"
                >
                  <rect
                    x={0}
                    y={0}
                    width={bed.w}
                    height={bed.d}
                    fill="var(--plate-ground)"
                    stroke="var(--line)"
                    strokeWidth={0.03}
                  />
                  {result.plants.map((p, k) => (
                    <circle
                      key={k}
                      cx={p.x}
                      cy={bed.d - p.y}
                      r={p.r}
                      fill={p.color}
                      fillOpacity={0.85}
                      stroke={p.pinned ? "var(--ink)" : "var(--line)"}
                      strokeWidth={p.pinned ? 0.04 : 0.015}
                    >
                      <title>
                        {p.species} ({p.layer})
                      </title>
                    </circle>
                  ))}
                </svg>
              </div>
              <div className="aux" style={{ marginTop: 6 }}>
                front of bed at the bottom · {result.plants.length} plants · pinned
                plants outlined
              </div>
              <div className="chips" style={{ marginTop: 10 }}>
                {Object.entries(result.metrics).map(([k, v]) => (
                  <span key={k} className="chip">
                    {k} {v.toFixed(2)}
                  </span>
                ))}
              </div>
              {result.note && <div className="note warn">{result.note}</div>}
              {result.ignored_pins.length > 0 && (
                <div className="note">
                  not in the palette: {result.ignored_pins.join(", ")}
                </div>
              )}
              <div className="actions">
                <button className="btn ghost" onClick={startRender} disabled={!!renderState}>
                  {renderState ? "Rendering…" : "Styled render (≈10–17 min)"}
                </button>
              </div>
              {renderState && <div className="note">{renderState}</div>}
              {renderImg && (
                <div className="preview" style={{ marginTop: 10 }}>
                  <img src={renderImg} alt="styled render of this plan" />
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <div className="card" style={{ marginTop: 20 }}>
        <h4>
          What the render looks like{" "}
          <span className="aux">pre-rendered examples · SD 1.5 + seg ControlNet (notebook 13)</span>
        </h4>
        <div className="grid2">
          <figure style={{ margin: 0 }}>
            <img src="/gallery/render_iter3.png" alt="styled render, iteration-3 plan"
                 style={{ width: "100%", borderRadius: 10 }} />
            <figcaption className="aux">a generated plan, rendered as a watercolor planting plan</figcaption>
          </figure>
          <figure style={{ margin: 0 }}>
            <img src="/gallery/cond_iter3.png" alt="conditioning image for the render"
                 style={{ width: "100%", borderRadius: 10 }} />
            <figcaption className="aux">its conditioning image — the plan is already a segmentation map</figcaption>
          </figure>
        </div>
      </div>
    </>
  );
}
