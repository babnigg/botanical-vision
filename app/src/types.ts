export type Status = "planned" | "candidate" | "live";
export type ModuleKey = "identify" | "illustrate" | "compose";
export type ModuleState = "live" | "partial" | "prototype";

export interface ModelEntry {
  id: string;
  name: string;
  module: ModuleKey;
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
