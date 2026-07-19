import { useStore } from "../store";
import TaxonTree from "../components/TaxonTree";

export default function Toolbox() {
  const { toolbox } = useStore();

  if (!toolbox.length)
    return (
      <div className="card">
        <div className="empty">
          Nothing collected yet. Identify a species and add it — favorites gather here and feed the
          garden studio.
        </div>
      </div>
    );

  return (
    <>
      <section className="hero" style={{ paddingBottom: 12 }}>
        <p className="kick">Shared across every module</p>
        <h1 style={{ fontSize: 28 }}>Your toolbox</h1>
      </section>
      <div className="grid2">
        <div className="card">
          <h4>
            Specimens <span className="aux">{toolbox.length}</span>
          </h4>
          <div className="tbgrid">
            {toolbox.map((e) => {
              const t = e.tax;
              return (
                <div className="tbcard" key={t.species}>
                  <div className="b">{t.species}</div>
                  <small>{t.common}</small>
                  <div className="chips">
                    <span className="chip">{t.genus}</span>
                    <span className="chip">{t.family}</span>
                    <span className="chip">IUCN {t.iucn}</span>
                  </div>
                  <div className="status">
                    <span className="s on">identified</span>
                    <span className={`s ${e.planted ? "on" : ""}`}>planted</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="card">
          <h4>
            Your collection as a tree <span className="aux">learning view</span>
          </h4>
          <TaxonTree items={toolbox.map((e) => e.tax)} />
          <div className="note">
            Every specimen you collect slots into the plant tree of life — see which families and
            genera your garden spans.
          </div>
        </div>
      </div>
    </>
  );
}
