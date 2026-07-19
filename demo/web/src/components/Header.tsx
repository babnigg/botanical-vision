function Leaf({ x, y, rot, s = 1 }: { x: number; y: number; rot: number; s?: number }) {
  return (
    <path
      className="leaf"
      transform={`translate(${x} ${y}) rotate(${rot}) scale(${s})`}
      d="M0 0 C 8 -6, 18 -3, 22 4 C 14 6, 5 4, 0 0 Z"
    />
  );
}

function Bloom({ cx, cy, r = 10 }: { cx: number; cy: number; r?: number }) {
  const petals = Array.from({ length: 10 }, (_, i) => {
    const a = (i / 10) * Math.PI * 2;
    const px = cx + Math.cos(a) * r;
    const py = cy + Math.sin(a) * r;
    return (
      <ellipse
        key={i}
        className="petal"
        cx={px}
        cy={py}
        rx={r * 0.5}
        ry={r * 0.22}
        transform={`rotate(${(a * 180) / Math.PI} ${px} ${py})`}
      />
    );
  });
  return (
    <g>
      {petals}
      <circle className="disc" cx={cx} cy={cy} r={r * 0.4} />
    </g>
  );
}

// One damask sprig: a curling vine with leaves, a bloom, and a couple of berries.
function Sprig({ t }: { t?: string }) {
  return (
    <g transform={t}>
      <path className="vine" d="M28 98 C 52 80, 16 60, 46 42 C 64 31, 90 40, 95 58" />
      <path className="vine" d="M95 58 C 99 46, 95 34, 86 28" />
      <Leaf x={42} y={72} rot={-42} s={0.85} />
      <Leaf x={28} y={56} rot={150} s={0.7} />
      <Leaf x={68} y={46} rot={-22} s={0.85} />
      <Leaf x={80} y={62} rot={175} s={0.7} />
      <Bloom cx={86} cy={22} r={10} />
      <circle className="berry" cx={52} cy={88} r={2} />
      <circle className="berry" cx={58} cy={92} r={2} />
    </g>
  );
}

// A tiled botanical damask band, tinted to the active theme.
export default function Header() {
  return (
    <div className="masthead" aria-hidden="true">
      <svg viewBox="0 0 1200 110" preserveAspectRatio="xMidYMid slice">
        <defs>
          <pattern id="bv-damask" width="180" height="110" patternUnits="userSpaceOnUse">
            <g className="damask">
              <Sprig />
              <Sprig t="translate(90 55)" />
              <Sprig t="translate(90 -55)" />
              <Sprig t="translate(-90 55)" />
            </g>
          </pattern>
        </defs>
        <rect width="1200" height="110" fill="url(#bv-damask)" />
      </svg>
    </div>
  );
}
