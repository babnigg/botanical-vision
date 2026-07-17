import { useStore } from "../store";
import type { ModuleKey } from "../types";

const MOD_INFO: Record<ModuleKey, { title: string; desc: string }> = {
  identify: { title: "Identify", desc: "Photo → species, with a top-5 and taxonomy." },
  illustrate: { title: "Illustrate", desc: "Species → a botanical plate in a chosen style." },
  compose: { title: "Compose", desc: "Site → beds → planting plan → styled render." },
};

export default function Overview() {
  const { registry, setView } = useStore();
  if (!registry)
    return (
      <div className="card">
        Connecting to the API at <code>localhost:8000</code>…
      </div>
    );

  return (
    <>
      <section className="hero">
        <h1>Project status</h1>
        <p>
          Each module runs on a model. This page reads what's actually deployed — nothing here is a
          switch. A model is <b>live</b> when its file is present and promoted to production,{" "}
          <b>candidate</b> when present but not chosen, and <b>planned</b> until it exists. Deploy a
          model and its module turns on by itself.
        </p>
      </section>

      <div className="pipeline">
        {(Object.keys(MOD_INFO) as ModuleKey[]).map((k) => {
          const st = registry.modules[k];
          const models = registry.models.filter((m) => m.module === k && !m.stretch);
          const done = models.filter((m) => m.status === "live").length;
          return (
            <div className="pstage" key={k} onClick={() => setView(k)} title={`Open ${MOD_INFO[k].title}`}>
              <h3>{MOD_INFO[k].title}</h3>
              <p>{MOD_INFO[k].desc}</p>
              <span className={`pill ${st}`}>
                <span className="d" />
                {st === "live" ? "Live" : st === "partial" ? "In progress" : "Not built yet"} · {done}/
                {models.length} models
              </span>
            </div>
          );
        })}
      </div>

      <div className="grid2">
        <div className="card">
          <h4>Models</h4>
          <div style={{ overflowX: "auto" }}>
            <table className="reg">
              <tbody>
                {registry.models.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <div className="mname">
                        {m.name}
                        {m.stretch ? " · stretch" : ""}
                      </div>
                      <div className="mmeta">
                        {MOD_INFO[m.module].title}
                        {m.metric ? ` · ${m.metric}` : ""}
                      </div>
                    </td>
                    <td>
                      <span className={`stbadge ${m.status}`}>
                        <span className="d" />
                        {m.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mono" style={{ fontSize: 11, color: "var(--ink-soft)", marginTop: 10 }}>
            {registry.live_count} of {registry.total} deployed
          </p>
        </div>

        <div className="card">
          <h4>Deploying a model</h4>
          <ol
            style={{
              margin: 0,
              paddingLeft: 18,
              fontSize: 13.5,
              color: "var(--ink-soft)",
              lineHeight: 1.75,
            }}
          >
            <li>Train it in the notebooks (runs are tracked in MLflow).</li>
            <li>
              Register it:{" "}
              <code>python -m mlops.register_checkpoint --checkpoint …</code> — it's
              packaged as an MLflow pyfunc model and versioned.
            </li>
            <li>
              Run the gate:{" "}
              <code>python -m mlops.promote --challenger &lt;version&gt;</code>. It's
              promoted to <code>@production</code> only if it beats the current champion
              on the holdout.
            </li>
            <li>
              The API serves the <code>@production</code> version automatically — no app
              code changes.
            </li>
          </ol>
          <div className="note">
            Status here is read from the <b>MLflow Model Registry</b>, not a file — a version is{" "}
            <b>live</b> when it holds the <code>production</code> alias. Promotion runs through a
            champion/challenger evaluation gate, so a worse model can't ship. Details in{" "}
            <code>models/README.md</code>.
          </div>
        </div>
      </div>
    </>
  );
}
