import { useStore, type ViewKey } from "../store";
import type { ModuleKey } from "../types";

const TABS: { key: ViewKey; label: string; mod?: ModuleKey }[] = [
  { key: "identify", label: "Identify", mod: "identify" },
  { key: "illustrate", label: "Illustrate", mod: "illustrate" },
  { key: "compose", label: "Compose", mod: "compose" },
  { key: "toolbox", label: "Toolbox" },
  { key: "overview", label: "Roadmap" },
];

export default function TopBar() {
  const { view, setView, theme, toggleTheme, registry, toolbox } = useStore();
  const mods = registry?.modules;
  return (
    <header className="topbar">
      <div className="img" aria-hidden="true" />
      <div className="scrim" aria-hidden="true" />
      <div className="inner">
        <div className="mark">
          <b>Botanical Vision</b>
        </div>
        <nav className="nav" aria-label="Modules">
          {TABS.map((t) => (
            <button key={t.key} aria-current={view === t.key} onClick={() => setView(t.key)}>
              {t.mod && <span className={`st ${mods?.[t.mod] ?? ""}`} />}
              {t.label}
              {t.key === "toolbox" && (
                <span className="mono" style={{ opacity: 0.6 }}>
                  {toolbox.length}
                </span>
              )}
            </button>
          ))}
        </nav>
        <div className="rightbar">
          <button className="buildchip" onClick={() => setView("overview")}>
            <b>{registry?.live_count ?? "—"}</b>/{registry?.total ?? 6} live
          </button>
          <button
            className="themetoggle"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
          >
            {theme === "dark" ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
                <circle cx="12" cy="12" r="4.4" />
                <path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5 5l1.6 1.6M17.4 17.4L19 19M19 5l-1.6 1.6M6.6 17.4L5 19" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.6 6.6 0 0 0 9.8 9.8Z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
