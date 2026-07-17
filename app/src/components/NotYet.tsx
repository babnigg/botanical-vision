export default function NotYet({
  title,
  needs,
  onRoadmap,
}: {
  title: string;
  needs: string;
  onRoadmap: () => void;
}) {
  return (
    <div className="card notyet">
      <h3>{title} isn't available yet</h3>
      <p>
        It runs on {needs}, which hasn't been trained and deployed. This section turns on
        automatically once that model is promoted in the registry — nothing else to wire.
      </p>
      <button className="btn ghost" onClick={onRoadmap}>
        See the roadmap
      </button>
    </div>
  );
}
