import { create } from "zustand";
import { api } from "./api";
import type { ModelInfo, ModuleKey, Prediction, RegistrySummary } from "./types";

export type ViewKey = "overview" | ModuleKey | "toolbox";
type Theme = "light" | "dark";

export interface ToolboxEntry {
  tax: Prediction; // carries species + full taxonomy
  planted?: boolean;
}

interface AppState {
  theme: Theme;
  toggleTheme: () => void;

  view: ViewKey;
  setView: (v: ViewKey) => void;

  registry?: RegistrySummary;
  refreshRegistry: () => Promise<void>;

  // model picker (lives on the Roadmap page)
  models: ModelInfo[];
  activeModelId?: string;
  modelsLoading: boolean;
  selectingId?: string;
  modelError?: string;
  fetchModels: () => Promise<void>;
  selectModel: (id: string) => Promise<void>;

  current?: Prediction; // the traveling specimen
  setCurrent: (p: Prediction) => void;

  toolbox: ToolboxEntry[];
  addToToolbox: (p: Prediction, opts?: { planted?: boolean }) => void;
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

  models: [],
  activeModelId: undefined,
  modelsLoading: false,
  selectingId: undefined,
  modelError: undefined,
  fetchModels: async () => {
    if (get().modelsLoading) return; // dedupe concurrent fetches (e.g. StrictMode)
    set({ modelsLoading: true });
    try {
      const r = await api.models();
      set({ models: r.models, activeModelId: r.active_id ?? undefined });
    } catch {
      set({ modelError: "Couldn't reach the API — is the backend running?" });
    } finally {
      set({ modelsLoading: false });
    }
  },
  selectModel: async (id) => {
    set({ selectingId: id, modelError: undefined });
    try {
      const r = await api.selectModel(id);
      if (!r.ok) set({ modelError: "That model couldn't be loaded — try another." });
      else set({ activeModelId: r.active_id ?? undefined });
      await get().fetchModels();
      await get().refreshRegistry().catch(() => {});
    } catch {
      set({ modelError: "Selection failed — is the API running?" });
    } finally {
      set({ selectingId: undefined });
    }
  },

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
              ? { ...t, planted: opts?.planted ?? t.planted }
              : t
          ),
        };
      }
      return { toolbox: [...s.toolbox, { tax: p, planted: opts?.planted }] };
    }),
}));
