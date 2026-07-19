import { useEffect } from "react";
import { useStore } from "./store";
import TopBar from "./components/TopBar";
import Overview from "./views/Overview";
import Identify from "./views/Identify";
import Compose from "./views/Compose";
import Toolbox from "./views/Toolbox";

const VIEWS = {
  overview: Overview,
  identify: Identify,
  compose: Compose,
  toolbox: Toolbox,
};

export default function App() {
  const { theme, view, refreshRegistry, fetchModels } = useStore();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    refreshRegistry().catch(() => {});
    fetchModels().catch(() => {});
  }, [refreshRegistry, fetchModels]);

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
