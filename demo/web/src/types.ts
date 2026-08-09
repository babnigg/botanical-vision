export type Status = "planned" | "candidate" | "live";
export type ModuleKey = "identify" | "compose";
export type ModuleState = "live" | "partial" | "prototype";

export interface ModelEntry {
  id: string;
  name: string;
  module: ModuleKey | "stylize";
  task?: string;
  base?: string;
  metric?: string;
  status: Status;
  stage?: string;
  stretch?: boolean;
  deployed?: boolean;
  notes?: string;
}

export interface RegistrySummary {
  models: ModelEntry[];
  modules: Record<ModuleKey, ModuleState>;
  live_count: number;
  total: number;
}

export interface Prediction {
  species: string;
  common: string;
  genus: string;
  family: string;
  order: string;
  iucn: string;
  confidence: number;
}

export interface ClassifyResponse {
  live: boolean;
  served?: "model" | "stub";
  reason?: string | null;
  model: string | null;
  predictions: Prediction[];
  taxonomy_agreement: {
    family: string;
    family_count: number;
    genus: string;
    genus_count: number;
  };
  note: string | null;
  module_status: ModuleState;
  mask_png_b64?: string | null;
  segmenter?: string | null;
}

export interface SpeciesReference {
  species: string;
  common: string;
  image: string | null;
  summary: string | null;
  links: { gbif?: string; inaturalist?: string; wikipedia?: string };
}

export interface ArrangePlant {
  sp: string;
  sun: string;
  height: number;
  bloom: [number, number];
  color: string;
  role: string;
  note: string;
  why: string;
  drift: number;
}

export interface ArrangeResponse {
  aspect: string;
  zone: string;
  plants: ArrangePlant[];
  bloom_months: number[];
  live: boolean;
  note: string | null;
}

export interface PaletteEntry {
  idx: number;
  name: string;
  common?: string;
  layer: string;
  h: number;
  s: number;
  sun: number;
  color: string;
}

export interface ComposePlant {
  species: string;
  common?: string;
  layer: string;
  color: string;
  h: number;
  x: number;
  y: number;
  r: number;
  pinned: boolean;
}

export interface ComposeResponse {
  live: boolean;
  served: "diffusion" | "rules";
  bed: { w: number; d: number; sun: number; sun_name: string };
  plants: ComposePlant[];
  metrics: Record<string, number>;
  ignored_pins: string[];
  note?: string;
}

export interface ModelInfo {
  id: string;
  source: "local" | "shared";
  name: string;
  species: number | null;
  val_acc: number | null;
  detail?: string;
  active?: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  active_id: string | null;
}
