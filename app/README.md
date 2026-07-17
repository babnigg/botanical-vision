# Botanical Vision — App (Vite + React + TypeScript)

The frontend. Component-based so it stays extensible as Compose grows. It talks
to the FastAPI backend through a dev proxy, so all calls are same-origin
`/api/*`. State lives in a small Zustand store (theme, current view, registry,
the shared toolbox, and the traveling specimen).

## Run

Start the **API first** (see `../api/README.md`), then:

```bash
cd app
npm install
npm run dev            # http://localhost:5173
```

`vite.config.ts` proxies `/api` → `http://localhost:8000`, so the two dev servers
work together with no CORS fuss.

## What's wired

- **Overview** — the roadmap. Live model registry with route-up (planned →
  training → live); flipping a status re-derives each module's live/prototype
  banner. Backed by `GET/POST /api/registry`.
- **Identify** — `POST /api/classify` → top-5 with confidence, taxonomy-aware
  agreement, and a **TaxonTree** (order → family → genus → species) that
  highlights your pick. Adds to the shared toolbox; hands the specimen to
  Illustrate.
- **Illustrate** — `POST /api/illustrate` builds the prompt from the shared vocab;
  render is a stub until the style LoRA is deployed.
- **Compose** — `POST /api/arrange` fills a bed from the seeded trait table
  (seeded by your toolbox), with a bloom calendar. The edge-aligned bed **canvas**
  and the ControlNet **plan render** port from the mockup next.
- **Toolbox** — the shared specimen collection, plus your collection rendered as a
  **TaxonTree** (the learning view).

## Layout

```
app/src/
  main.tsx, App.tsx, styles.css   # entry, shell, herbarium/cyanotype tokens
  types.ts                        # mirrors the API responses
  api.ts                          # typed fetch client
  store.ts                        # Zustand store
  components/
    TopBar.tsx
    TaxonTree.tsx                 # reused by Identify + Toolbox
  views/
    Overview.tsx  Identify.tsx  Illustrate.tsx  Compose.tsx  Toolbox.tsx
```

## Design language

Two botanical printing traditions, one system: **light = herbarium sheet**,
**dark = cyanotype**. Tokens live at the top of `styles.css`; the theme toggle
stamps `data-theme` on `<html>`.

## Next

Port the mockup's Compose canvas (edge-aligned beds, sketch-your-own, styled plan
render) and the plate rendering into these components; add file-upload to Identify
and the SAM subject-segmenter (`subject-segmenter` registry row) as the Module-1
stretch.
