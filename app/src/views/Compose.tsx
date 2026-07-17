import { useStore } from "../store";
import NotYet from "../components/NotYet";

export default function Compose() {
  const { registry, setView } = useStore();
  const live = (registry?.modules.compose ?? "prototype") === "live";

  if (!live)
    return (
      <NotYet
        title="Garden design"
        needs="a plant trait table, an arrangement engine, and a ControlNet render model"
        onRoadmap={() => setView("overview")}
      />
    );

  // When the Compose models are deployed, the garden studio goes here.
  return (
    <div className="card">
      <div className="empty">Compose models deployed — the garden studio will live here.</div>
    </div>
  );
}
