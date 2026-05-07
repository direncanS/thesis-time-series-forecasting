# RENDER_SPEC

Thesis render/export spec. Locked originally (2026-04-20) with Gate B and Gate C decisions.

## Gate B decision (render format + toolchain)

- **Output format:** PDF (FH Joanneum default)
- **Toolchain:** Pandoc (installed in `thesis` conda env via conda-forge 2026-04-20)
- **Citation strategy:** (i) **markdown-assembly** — default, no `.bib` extraction. Harvard in-text citations are already plain-text in `docs/*.md`. The `docs/references.md` section is concatenated into the PDF as the bibliography.
- **PDF engine:** **tectonic** (installed via conda-forge; self-contained LaTeX engine with full Unicode support — handles German umlauts ä ö ü ß natively)
- **Thesis language:** German (user decision 2026-04-20). `-V lang=de` flag required.

## Gate C decision (render failure policy)

- **Primary:** **A — Dış makinede render.** Transportable package on render failure: `docs/*.md` (7 prose files) + `docs/RENDER_SPEC.md` + command below. No `.bib` / `.csl` needed on default path.
- Fallback within A: online Pandoc services (pandoc-try.ofmain.io or similar) can be used if local install impossible on secondary machine.

## Merge order

```
docs/abstract.md
docs/introduction.md
docs/methodology.md
docs/solution.md
docs/discussion.md
docs/references.md
docs/appendix.md
```

## Canonical render command (Windows PowerShell)

```powershell
pandoc `
 docs/abstract.md docs/introduction.md docs/methodology.md `
 docs/solution.md docs/discussion.md docs/references.md docs/appendix.md `
 --pdf-engine=tectonic `
 --toc --toc-depth=3 `
 -V geometry:margin=2.5cm `
 -V mainfont="Times New Roman" `
 -V lang=de `
 -o docs/thesis_final.pdf
```

**Bash / Git Bash equivalent:**

```bash
pandoc \
 docs/abstract.md docs/introduction.md docs/methodology.md \
 docs/solution.md docs/discussion.md docs/references.md docs/appendix.md \
 --pdf-engine=tectonic \
 --toc --toc-depth=3 \
 -V geometry:margin=2.5cm \
 -V mainfont="Times New Roman" \
 -V lang=de \
 -o docs/thesis_final.pdf
```

## Output

- **Path:** `docs/thesis_final.pdf`
- **Expected size:** ~15–30 pages of main body + appendix; ~1–3 MB PDF.

## Minimal render smoke test

Smoke subset — just abstract + introduction + references (tiny, fast, no appendix images):

```powershell
pandoc docs/abstract.md docs/introduction.md docs/references.md `
 --pdf-engine=tectonic `
 -o docs/_smoke_render.pdf
```

**Success criterion:** Command returns exit code 0 and `docs/_smoke_render.pdf` exists and opens.

If xelatex is not installed locally, Gate C fallback A activates (transport docs/*.md to external machine).

## Prerequisites (Gate D — DONE in 2026-04-20)

Installed in `thesis` conda env via conda-forge:

```powershell
conda install -n thesis -c conda-forge pandoc tectonic -y
```

Env-scoped (no admin), ~200 MB total. Alternative system-wide paths (winget / scoop / MiKTeX) remain documented in plan v11 but were not taken.

## Fallback package for "A — dış makine" scenario

Transport this minimal set to the external machine:

```
docs/abstract.md
docs/introduction.md
docs/methodology.md
docs/solution.md
docs/discussion.md
docs/references.md
docs/appendix.md
docs/RENDER_SPEC.md
```

No `.bib`, no `.csl`, no `images/` (unless appendix figures are added in later sessions — if so, include `docs/images/*` in the transport package).

## Render log (2026-04-20)

- **Date:** 2026-04-20
- **Toolchain install:** `conda install -n thesis -c conda-forge pandoc tectonic -y` — success.
- **Pandoc version:** conda-forge (exact version in `conda list` output; installed 2026-04-20).
- **xelatex / MiKTeX:** NOT installed (user chose tectonic via Seçenek A, env-scoped lightweight path).
- **PDF engine actually used:** tectonic
- **Smoke render command run:**
 ```
 conda run -n thesis pandoc docs/abstract.md docs/introduction.md docs/references.md --pdf-engine=tectonic -V lang=de -o docs/_smoke_render.pdf
 ```
- **Smoke render `docs/_smoke_render.pdf` produced:** YES — 28,707 bytes, created 2026-04-20 23:55.
- **Fallback A triggered:** NO — local render works.
- **Notes:** Thesis language set to German per user decision; `-V lang=de` baked into canonical command. Existing prose in `docs/*.md` currently mixed English/Turkish; final German translation is a separate workitem (not part of ).
