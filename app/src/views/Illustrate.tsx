import { useStore } from "../store";
import NotYet from "../components/NotYet";

export default function Illustrate() {
  const { registry, setView } = useStore();
  const live = (registry?.modules.illustrate ?? "prototype") === "live";

  if (!live)
    return (
      <NotYet
        title="Illustration"
        needs="a botanical-plate style model (Stable Diffusion + a trained style LoRA)"
        onRoadmap={() => setView("overview")}
      />
    );

  // When a style model is deployed, the illustrator UI goes here.
  return (
    <div className="card">
      <div className="empty">Style model deployed — the illustrator UI will live here.</div>
    </div>
  );
}
