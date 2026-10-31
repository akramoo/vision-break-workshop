# Build a Vision Model, Then Break It

A browser-only teaching app for a two-day university workshop (1-2 Nov 2026): train a pneumonia
classifier on children's chest X-rays with frozen MobileNet v2 features, then break it with class
imbalance, shortcuts, population shift, image-quality changes and out-of-distribution input.

**Educational only. Not for clinical use.**

Status: under construction. See [PLAN.md](PLAN.md) for the build plan and [WORKSHOP.md](WORKSHOP.md)
for the teaching plan. Full run, deploy and offline-USB instructions will be added here.

- `app/` - Vite + React + TypeScript app
- `scripts/` - data preparation (Python) and embedding precompute (Node)
- `notebook/` - optional Python code track
- `docs/` - facilitator notes and dry-run results

Licensed under MIT. Data sources and their licenses: see DATA_LICENSES.md (coming with step D4).
