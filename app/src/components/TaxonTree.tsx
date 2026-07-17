import type { Prediction } from "../types";

type Item = Pick<Prediction, "species" | "genus" | "family" | "order"> & {
  confidence?: number;
};

interface Node {
  rank: "order" | "family" | "genus" | "species";
  name: string;
  children: Node[];
  leaf?: Item;
}

function build(items: Item[]): Node[] {
  const roots: Node[] = [];
  const find = (arr: Node[], rank: Node["rank"], name: string): Node => {
    let n = arr.find((x) => x.rank === rank && x.name === name);
    if (!n) {
      n = { rank, name, children: [] };
      arr.push(n);
    }
    return n;
  };
  for (const it of items) {
    const o = find(roots, "order", it.order || "—");
    const f = find(o.children, "family", it.family || "—");
    const g = find(f.children, "genus", it.genus || "—");
    g.children.push({ rank: "species", name: it.species, children: [], leaf: it });
  }
  return roots;
}

function Nodes({ nodes, highlight }: { nodes: Node[]; highlight?: string }) {
  return (
    <ul>
      {nodes.map((n, i) => (
        <li key={`${n.rank}-${n.name}-${i}`}>
          <span className={`node ${n.rank}${n.rank === "species" && n.name === highlight ? " hit" : ""}`}>
            <span className="rank">{n.rank}</span>
            <span className="rname">{n.name}</span>
            {n.leaf?.confidence != null && (
              <span className="cf">{n.leaf.confidence.toFixed(1)}%</span>
            )}
          </span>
          {n.children.length > 0 && <Nodes nodes={n.children} highlight={highlight} />}
        </li>
      ))}
    </ul>
  );
}

export default function TaxonTree({
  items,
  highlight,
}: {
  items: Item[];
  highlight?: string;
}) {
  if (!items.length) return null;
  return (
    <div className="tree">
      <Nodes nodes={build(items)} highlight={highlight} />
    </div>
  );
}
