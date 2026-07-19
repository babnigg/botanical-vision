import { useEffect, useRef, useState, type CSSProperties } from "react";
import { api } from "../api";
import { useStore } from "../store";
import type { ClassifyResponse, Prediction, SpeciesReference } from "../types";
import TaxonTree from "../components/TaxonTree";

export default function Identify() {
  const { current, setCurrent, addToToolbox, models, activeModelId, setView } = useStore();
  const activeModel = models.find((m) => m.id === activeModelId);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [remoteUrl, setRemoteUrl] = useState("");
  const [res, setRes] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [randomLoading, setRandomLoading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [torch, setTorch] = useState<boolean | null>(null);
  const [ref, setRef] = useState<SpeciesReference | null>(null);
  const [refLoading, setRefLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((d) => setTorch(!!d.torch)).catch(() => setTorch(null));
  }, []);
  useEffect(() => () => { if (preview.startsWith("blob:")) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    if (!current) { setRef(null); return; }
    let cancelled = false;
    setRef(null);
    setRefLoading(true);
    api
      .reference(current.species)
      .then((r) => { if (!cancelled) setRef(r); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setRefLoading(false); });
    return () => { cancelled = true; };
  }, [current?.species]);

  function swapPreview(next: string) {
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setPreview(next);
  }
  function chooseFile(f: File | null) {
    if (!f) return;
    setFile(f);
    setRemoteUrl("");
    setRes(null);
    swapPreview(URL.createObjectURL(f));
  }
  async function pullRandom() {
    setRandomLoading(true);
    try {
      const r = await api.randomPhoto();
      if (r.url) {
        setFile(null);
        setRemoteUrl(r.url);
        setRes(null);
        swapPreview(r.url);
      }
    } finally {
      setRandomLoading(false);
    }
  }
  function clearPhoto() {
    setFile(null);
    setRemoteUrl("");
    setRes(null);
    swapPreview("");
  }

  async function run() {
    if (!file && !remoteUrl) return;
    setLoading(true);
    try {
      const r = await api.classify({ file: file ?? undefined, url: file ? undefined : remoteUrl || undefined });
      setRes(r);
      setCurrent(r.predictions[0]);
    } finally {
      setLoading(false);
    }
  }

  const hasPhoto = !!file || !!remoteUrl;
  const isStub = res?.served === "stub";

  const linkBtn: CSSProperties = {
    background: "none", border: "none", padding: 0, font: "inherit",
    color: "inherit", textDecoration: "underline", cursor: "pointer",
  };

  return (
    <>
      {activeModelId ? (
        <div className="note" style={{ marginTop: 0, marginBottom: 18 }}>
          Serving <b>{activeModel?.name ?? activeModelId}</b>
          {activeModel ? ` · ${activeModel.species} species` : ""}.{" "}
          <button style={linkBtn} onClick={() => setView("overview")}>change on the Roadmap</button>
        </div>
      ) : (
        <div className="note warn" style={{ marginTop: 0, marginBottom: 18 }}>
          No model selected.{" "}
          <button style={linkBtn} onClick={() => setView("overview")}>Choose one on the Roadmap →</button>
        </div>
      )}
      {torch === false && (
        <div className="note warn" style={{ marginTop: 0, marginBottom: 18 }}>
          The classifier is deployed, but <b>PyTorch isn't installed here</b>, so photos can't be run
          through the model yet — you'll get a demo result. Install it with{" "}
          <code>pip install torch torchvision</code> to identify real images.
        </div>
      )}

      <div className="grid2">
        <div className="card">
          <h4>Your photograph</h4>

          {!hasPhoto ? (
            <div
              className={`dropzone${drag ? " drag" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); chooseFile(e.dataTransfer.files?.[0] ?? null); }}
            >
              <strong>Drop a photo here</strong>, or click to choose
              <div className="hint">JPG or PNG — a single plant, close up. It's resized to 224² for the model.</div>
            </div>
          ) : (
            <div className="preview">
              <img src={preview} alt="plant to identify" />
              <button className="clearbtn" onClick={clearPhoto} title="Remove photo" aria-label="Remove photo">×</button>
            </div>
          )}

          <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => chooseFile(e.target.files?.[0] ?? null)} />

          <div className="actions">
            <button className="btn" onClick={run} disabled={!hasPhoto || loading}>
              {loading ? "Identifying…" : "Identify species"}
            </button>
            <button className="btn ghost" onClick={pullRandom} disabled={randomLoading}>
              {randomLoading ? "Finding…" : "Random plant"}
            </button>
          </div>
        </div>

        <div className="card">
          <h4>
            Top matches{" "}
            <span className="aux">{res ? "— 5 of 4,094 species" : "— add a photo"}</span>
          </h4>

          {!res && (
            <div style={{ color: "var(--ink-soft)", fontSize: 13.5, padding: "20px 4px" }}>
              Add a plant photo and the classifier returns its five most likely species with
              confidence, plus how they sit in the taxonomy.
            </div>
          )}

          {res && isStub && (
            <div className="note warn" style={{ marginTop: 0, marginBottom: 12 }}>
              {res.note ?? "Demo result — not run on your image."}
            </div>
          )}

          {res && (
            <ul className="results">
              {res.predictions.map((p: Prediction, i) => (
                <li key={p.species} aria-selected={current?.species === p.species} onClick={() => setCurrent(p)}>
                  <span className="rk">{i + 1}</span>
                  <span>
                    <span className="binom">{p.species}</span>
                    {p.common && <span className="com"> · {p.common}</span>}
                    <div className="tax">
                      {[p.genus, p.family, p.order].filter(Boolean).join(" · ")}
                      {p.iucn && <span className={`iucn ${p.iucn}`}>{p.iucn}</span>}
                    </div>
                  </span>
                  <span className="conf">
                    <span className="pct num">{p.confidence.toFixed(1)}%</span>
                    <span className="bar"><i style={{ width: `${Math.max(2, p.confidence)}%` }} /></span>
                  </span>
                </li>
              ))}
            </ul>
          )}

          {res && res.taxonomy_agreement.family && (
            <div className="agg">
              Top-5 taxonomy: <b>{res.taxonomy_agreement.family_count}/5</b> share family{" "}
              <b>{res.taxonomy_agreement.family}</b>, <b>{res.taxonomy_agreement.genus_count}/5</b> share
              genus <b>{res.taxonomy_agreement.genus}</b>.
            </div>
          )}

          {res && (
            <div className="actions">
              <button className="btn ghost" onClick={() => current && addToToolbox(current)}>
                Add to toolbox
              </button>
            </div>
          )}
        </div>
      </div>

      {res && current && (
        <div className="card" style={{ marginTop: 20 }}>
          <h4>
            <span>
              About{" "}
              <span className="binom" style={{ textTransform: "none", letterSpacing: 0 }}>
                {current.species}
              </span>
            </span>
            {current.common && <span className="aux">{current.common}</span>}
          </h4>
          <div className="ref">
            {ref?.image ? (
              <img className="refimg" src={ref.image} alt={current.species} loading="lazy" />
            ) : (
              <div className="refimg placeholder">{refLoading ? "loading…" : "no image"}</div>
            )}
            <div className="reftext">
              {refLoading && !ref && <p style={{ color: "var(--ink-soft)" }}>Fetching a reference…</p>}
              {ref?.summary && <p>{ref.summary}</p>}
              {ref && !ref.summary && !refLoading && (
                <p style={{ color: "var(--ink-soft)" }}>
                  No description found — open a reference below to read more.
                </p>
              )}
              <div className="reflinks">
                {ref?.links.gbif && (
                  <a className="reflink" href={ref.links.gbif} target="_blank" rel="noreferrer">GBIF ↗</a>
                )}
                {ref?.links.inaturalist && (
                  <a className="reflink" href={ref.links.inaturalist} target="_blank" rel="noreferrer">iNaturalist ↗</a>
                )}
                {ref?.links.wikipedia && (
                  <a className="reflink" href={ref.links.wikipedia} target="_blank" rel="noreferrer">Wikipedia ↗</a>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {res && (
        <div className="card" style={{ marginTop: 20 }}>
          <h4>
            How the guesses relate{" "}
            <span className="aux">order → family → genus → species</span>
          </h4>
          <TaxonTree items={res.predictions} highlight={current?.species} />
        </div>
      )}
    </>
  );
}
