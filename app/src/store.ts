import { create } from "zustand";
import { api } from "./api";
import type { ModuleKey, Prediction, RegistrySummary } from "./types";

export type ViewKey = "overview" | ModuleKey | "toolbox";
type Theme = "light" | "dark";

export interface ToolboxEntry {
  tax: Prediction; // carries species + full taxonomy
  plate?: string; // style name if illustrated
  planted?: boolean;
}

interface AppState {
  theme: Theme;
  toggleTheme: () => void;

  view: ViewKey;
  setView: (v: ViewKey) => void;

  registry?: RegistrySummary;
  refreshRegistry: () => Promise<void>;

  current?: Prediction; // the traveling specimen
  setCurrent: (p: Prediction) => void;

  toolbox: ToolboxEntry[];
  addToToolbox: (p: Prediction, opts?: { plate?: string; planted?: boolean }) => void;
}

export const useStore = create<AppState>((set, get) => ({
  theme:
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light",
  toggleTheme: () =>
    set((s) => {
      const theme = s.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", theme);
      return { theme };
    }),

  view: "identify",
  setView: (view) => set({ view }),

  registry: undefined,
  refreshRegistry: async () => set({ registry: await api.registry() }),

  current: undefined,
  setCurrent: (current) => set({ current }),

  toolbox: [],
  addToToolbox: (p, opts) =>
    set((s) => {
      const existing = s.toolbox.find((t) => t.tax.species === p.species);
      if (existing) {
        return {
          toolbox: s.toolbox.map((t) =>
            t.tax.species === p.species
              ? { ...t, plate: opts?.plate ?? t.plate, planted: opts?.planted ?? t.planted }
              : t
          ),
        };
      }
      return { toolbox: [...s.toolbox, { tax: p, plate: opts?.plate, planted: opts?.planted }] };
    }),
}));
