import type {
  ArrangeResponse,
  ClassifyResponse,
  ModelsResponse,
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
