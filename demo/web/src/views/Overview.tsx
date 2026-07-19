import { useEffect } from "react";
import { useStore } from "../store";
import type { ModelInfo, ModuleKey } from "../types";

const MOD_INFO: Record<ModuleKey, { title: string; desc: string }> = {
  identify: { title: "Identify", desc: "Photo → species, with a top-5 and taxonomy." },
  compose: { title: "Compose", desc: "Site → beds → planting plan (garden design)." },
};

function ModelRow({ m }: { m: ModelInfo }) {
  const { activeModelId, selectingId, selectModel } = useStore();
  const meta = `${m.species ?? "?"} species${m.val_acc != null ? ` · val ${m.val_acc.toFixed(3)}` : ""} · ${m.source}`;
  return (
    <tr>
      <td>
        <div className="mname">{m.name}</div>
        <div className="mmeta">{meta}</div>
      </td>
      <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
        {m.id === activeModelId ? (
          <span className="stbadge live"><span className="d" />serving</span>
        ) : selectingId === m.id ? (
          <span className="aux">{m.source === "shared" ? "downloading…" : "loading…"}</span>
        ) : (
          <button className="btn ghost" disabled={!!selectingId} onClick={() => selectModel(m.id)}>
            Use
          </button>
        )}
      </td>
    </tr>
  );
}

export default function Overview() {
  const { registry, setView, models, activeModelId, modelsLoading, selectingId, modelError, fetchModels } =
    useStore();

  useEffect(() => {
    fetchModels().catch(() => {});
  }, [fetchModels]);

  return (
    <>
      <section className="hero">
        <h1>Project status</h1>
        <p>
          Each module runs on a model. <b>Identify</b> serves whichever model you pick below —
          nothing is auto-selected. <b>Compose</b> (garden design) is planned until its trait table
          and arrangement engine exist.
        </p>
      </section>

      <div className="pipeline">
        {(Object.keys(MOD_INFO) as ModuleKey[]).map((k) => {
          const st = registry?.modules[k] ?? "prototype";
          return (
            <div className="pstage" key={k} onClick={() => setView(k)} title={`Open ${MOD_INFO[k].title}`}>
              <h3>{MOD_INFO[k].title}</h3>
              <p>{MOD_INFO[k].desc}</p>
              <span className={`pill ${st}`}>
                <span className="d" />
                {st === "live" ? "Live" : st === "partial" ? "In progress" : "Not built yet"}
              </span>
            </div>
          );
        })}
      </div>

      <div className="grid2">
        <div className="card">
          <h4>
            Serving model{" "}
            <span className="aux">local checkpoints + shared — you pick which one Identify uses</span>
          </h4>

          {modelsLoading && models.length === 0 ? (
            <div className="empty">Loading models…</div>
          ) : models.length === 0 ? (
            <div className="empty">
              No models found. Train one in the notebooks, or publish one with <code>share.publish</code>.
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="reg">
                <tbody>
                  {models.map((m) => (
                    <ModelRow key={m.id} m={m} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {modelError && (
            <div className="note warn" style={{ marginTop: 10 }}>
              {modelError}
            </div>
          )}
          {!activeModelId && !selectingId && models.length > 0 && (
            <div className="note" style={{ marginTop: 10 }}>
              No model selected — pick one to enable Identify.
            </div>
          )}
        </div>

        <div className="card">
          <h4>How a model reaches the app</h4>
          <ol
            style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, color: "var(--ink-soft)", lineHeight: 1.75 }}
          >
            <li>Train it in the notebooks (03 / 04) → a checkpoint in <code>checkpoints/</code>.</li>
            <li>
              Share it with the team: <code>python -m share.publish --checkpoint … --name …</code>
            </li>
            <li>
              Compare everyone's on the same test split: <code>python -m share.leaderboard</code>
            </li>
            <li>Pick any of them here — local checkpoints and shared models both appear, with species + accuracy.</li>
          </ol>
          <div className="note">
            No registry, no promotion gate — you choose the served model explicitly. Identify shows{" "}
            <b>live</b> once a model is selected, otherwise a labeled stub.
          </div>
        </div>
      </div>
    </>
  );
}
