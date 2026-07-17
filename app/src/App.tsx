import { useEffect } from "react";
import { useStore } from "./store";
import TopBar from "./components/TopBar";
import Overview from "./views/Overview";
import Identify from "./views/Identify";
import Illustrate from "./views/Illustrate";
import Compose from "./views/Compose";
import Toolbox from "./views/Toolbox";

const VIEWS = {
  overview: Overview,
  identify: Identify,
  illustrate: Illustrate,
  compose: Compose,
  toolbox: Toolbox,
};

export default function App() {
  const { theme, view, refreshRegistry } = useStore();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    refreshRegistry().catch(() => {});
  }, [refreshRegistry]);

  const View = VIEWS[view];
  return (
    <div className="wrap">
      <TopBar />
      <main className="main">
        <View />
      </main>
    </div>
  );
}
