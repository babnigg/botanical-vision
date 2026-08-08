import type {
  ArrangeResponse,
  ClassifyResponse,
  ComposeResponse,
  ModelsResponse,
  ModuleState,
  PaletteEntry,
  RegistrySummary,
  SpeciesReference,
} from "./types";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export const api = {
  registry: () => fetch("/api/registry").then(json<RegistrySummary>),

  classify: (opts: { file?: File; url?: string }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.url) fd.append("url", opts.url);
    return fetch("/api/classify", { method: "POST", body: fd }).then(json<ClassifyResponse>);
  },

  arrange: (body: {
    aspect: string;
    zone: string;
    area?: number;
    toolbox?: string[];
  }) =>
    fetch("/api/arrange", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<ArrangeResponse>),

  zone: (zip: string) =>
    fetch(`/api/zone/${zip}`).then(json<{ zip: string; zone: string }>),

  composePalette: () =>
    fetch("/api/compose/palette").then(
      json<{ palette: PaletteEntry[]; status: ModuleState }>,
    ),

  compose: (body: {
    width: number;
    depth: number;
    sun: number;
    pins: { species: string; count: number }[];
  }) =>
    fetch("/api/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<ComposeResponse>),

  composeRender: (body: {
    width: number;
    depth: number;
    plants: { species: string; x: number; y: number; r: number }[];
  }) =>
    fetch("/api/compose/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<{ ok: boolean; job?: string; error?: string }>),

  composeRenderStatus: (job: string) =>
    fetch(`/api/compose/render/${job}`).then(
      json<{ status: string; elapsed?: number; png_b64?: string; error?: string }>,
    ),

  reference: (species: string) =>
    fetch(`/api/reference/${encodeURIComponent(species)}`).then(json<SpeciesReference>),

  randomPhoto: () =>
    fetch("/api/random-photo").then(json<{ url: string | null; species: string | null }>),

  models: () => fetch("/api/models").then(json<ModelsResponse>),

  selectModel: (id: string) =>
    fetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    }).then(json<{ ok: boolean; active_id: string | null }>),
};
