# Little Figures Studio 3.1

Turn one photo into an original rounded figurine scene of the whole group. The photo supplies poses, broad clothing shapes, and recognized accessories. Runs locally without an account, paid API, or photo upload to a model provider.

**[Download the Windows app](https://github.com/kevinhoangkim-netizen/smiski3D/raw/refs/heads/main/downloads/Little-Figures-Studio-Windows.zip?version=3.1)**

## Windows

For an existing installation, [download the small updater ZIP](https://github.com/kevinhoangkim-netizen/smiski3D/raw/refs/heads/main/downloads/Little-Figures-Updater.zip), extract its one file, **Update Little Figures.bat**, beside **Start Little Figures.bat**, and double-click it. Close the old launcher before this first update. Later updates can stop/restart the app automatically when analysis/generation is idle. Your models, Python environment, projects, and browser data stay in place. Save important projects before restarting.

For a new installation:

1. Extract the entire Windows ZIP into its own writable folder.
2. Double-click **Start Little Figures.bat**. Install Python 3.12 if prompted, then reopen it.
3. Keep the launcher open during first setup. It downloads dependencies and roughly 255 MiB of local vision weights, then opens your browser.
4. Use **Update Little Figures.bat** for later releases. No Git or full ZIP extraction is needed.

The updater verifies the HTTPS release archive and each file before changing the app, backs up changed files under `.local/updates/backups`, and recovers interrupted file updates. `scripts/update_local.py --check` checks without applying; `--rollback --no-launch` restores the most recent app-file backup. Dependencies and models are retained; this rollback concerns app files.
The compiled website is included: Node.js is unnecessary. Later launches reuse verified downloads. Existing version-1 project JSON files remain viewable, with legacy preset scenes clearly labeled. Windows wheels are available for the native dependencies; native Windows execution has not been tested here. The launcher and app are tested on Linux CPU/Chromium.

## The photo-first flow

**Upload → review everyone → generate the group → inspect and download.**

There are no pose, outfit, or accessory menus, occasion presets, or manual figure placement controls. Startup is empty; it never silently generates a stock person. Select a person only to inspect their read-only details. Everyone remains in the scene and downloads together.

- **Independent people detection:** local YOLOX finds body regions before per-person MediaPipe pose analysis. Full-height crops, global candidates, duplicate suppression and pose quality checks improve natural group handling. Six people maximum.
- **Explicit completeness:** numbered outlines are labeled detection indicators, not crops. The count includes detected people without a usable pose. Those regions remain visible and block generation. Overflow also blocks partial output. Confirm everyone is included before generating.
- **Missing-person recovery:** choose “Someone is missing” and draw around their body. The same original photo is rechecked with the additional region; existing people stay in the analysis. A known person's pose cannot satisfy a request for an additional person without new evidence. Failed recovery remains unresolved.
- **Photo-based garments:** local SCHP ATR-18 segmentation identifies broad clothing regions; local CLIP adds supported cues such as jackets, bridal dresses, hats and bouquets. Coverage measurements inform sleeves, trousers/shorts, hems and skirt flare. These become solid geometry with garment hems, cuffs, necklines, jacket lapels/buttons, and skirt folds, following the estimated joints. Recognized flowers have modeled petals/leaves/ribbons; hats and bags have physical seams and closures. Fine patterns and exact cuts remain approximate.
- **Recognized accessories:** simplified hats, bouquets, veils, bow ties, bags, scarves and glasses. Unfamiliar objects and small/hidden details can be missed. Strongly recognized unsupported wheelchairs/canes block generation rather than disappearing silently; recognition of those objects is not guaranteed.
- **Consistent sculpt:** large round heads, small fused rounded noses, tiny recessed eyes/mouths, smooth bodies and limbs, small feet, pale green preview material. Back-facing shoulder order places the face and garment front on the opposite surface without mirroring limb positions. Original procedural geometry; no official character mesh or affiliation.
- **Photo pose mapping:** reliable visible joints keep their image-plane proportions and foreshortening instead of being stretched to fixed limb lengths. Hidden joints still use a labeled fallback; a single view cannot establish exact depth.
- **Print bases:** new photo groups start with fused round bases under each person. Choose round/rectangular figure bases or one shared pedestal, adjust thickness/margin, and regenerate. Individual base thickness stays in physical millimeters when resizing.
- **Whole scene controls:** orbit/zoom, full-group bounds, uniform sizing that preserves relative sizes/spacing, optional shared pedestal and inscription, print checks, JSON save/open, and photo-free browser draft recovery.
- **Sharing:** group STL, colored GLB, PNG, actual eight-second WebM, and a self-contained interactive HTML file. HTML recipients open the file in a WebGL-capable browser. No public hosting URL is created.

This is a stylized single-photo interpretation, not an exact scan or arbitrary garment/object reconstruction. Read uncertainty notes before printing. Unclear clothing is automatically estimated from available segmentation, similarity scores, measurements and known outfit context; missing evidence falls back to clearly labeled simple clothing. Estimates do not block generation. Unresolved people and major unsupported objects remain blockers. See [LIMITATIONS.md](LIMITATIONS.md).

## Printing and saving

The main STL contains every included person in the photo group. Without a pedestal, one file can contain separate closed figures. A shared pedestal can connect the group physically. Regenerate after size or pedestal changes; stale geometry cannot be exported.

Coordinates are **millimeters, Z up**. STL contains no units or colors: select millimeters in your slicer. GLB uses meters/Y up and preserves materials. Facial marks are real recesses; glow requires an appropriate physical material. Print checks do not create supports or guarantee print success.

Browser drafts and JSON store names, landmarks, clothing measurements and scene settings, never photo bytes or filenames. Clearing browser storage or changing local origin can hide a draft; save JSON for important projects. Removing the reference photo keeps the derived group. HTML/GLB contain figure names and geometry, not the original photo.

## Development

Baseline: Python 3.12, Node 24, Linux CPU, Chromium. No secrets or GPU required.

```bash
bash scripts/install.sh
bash scripts/dev.sh
```

Development defaults: FastAPI on 127.0.0.1:8000 and Vite on 5173, proxying `/api`. Override both with `LITTLE_FIGURES_BACKEND_PORT` and `LITTLE_FIGURES_FRONTEND_PORT`. Tests can select `CHROMIUM_PATH`. Optional model directory/path overrides are documented in `.env.example`; the backend does not automatically load `.env`.

Production-style local launch:

```bash
.venv/bin/python scripts/launch_local.py --no-browser --skip-install --port 8767
```

It serves compiled studio, `/welcome/`, and API together on an available loopback port. `--skip-install` is for an already prepared environment. The Windows launcher performs installation automatically. The launcher only reuses a server with the same application version. It records an authenticated local update control without exposing its token to the browser.

### Local models

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_appearance_model.py
.venv/bin/python scripts/download_people_model.py
.venv/bin/python scripts/download_garment_model.py
```

Installers download weights only, over verified HTTPS. Inference reads local files and does not download anything. Person and garment models use pinned revisions and SHA-256; CLIP verifies upstream model hashes and tokenizer Git object checksum, then rechecks local SHA-256 at runtime. Weights and test photos are excluded from the ZIP.

| Capability | Model | Approx. download | Source/license |
| --- | --- | --- | --- |
| Pose | MediaPipe Pose Landmarker Lite | 6 MiB | Google MediaPipe, Apache-2.0 |
| People | YOLOX-S quantized ONNX | 34 MiB | `amd/yolox-s`, Apache-2.0 |
| Clothing regions | SCHP ATR-18 quantized ONNX | 66 MiB | `pirocheto/schp-atr-18`, MIT; notice in `scripts/model_licenses/SCHP-MIT.txt` |
| Fine clothing/accessory cues | CLIP ViT-B/32 quantized ONNX | 149 MiB | `Xenova/clip-vit-base-patch32`, MIT |

Hosts: `storage.googleapis.com`, `huggingface.co`, and its download CDN `us.aws.cdn.hf.co`. Missing models are reported explicitly. Existing projects can still be opened; new photo generation must not masquerade as inferred output when analysis is unavailable.

### Checks

```bash
.venv/bin/python -m pytest -q
npm run build --prefix frontend
npm run test:e2e --prefix frontend
.venv/bin/python -m scripts.verify_photo /path/to/group.jpg --expected-people 3 --shared-base
.venv/bin/python scripts/package_windows.py
```

Python checks cover actual geometry/STL topology, measured garment variations, pose/aspect/orientation, detection deduplication and unresolved recovery, model integrity, complete-group review enforcement, bounded uploads and queues. Browser checks cover empty startup, absence of manual selectors, whole-group generation, additive recovery, clothing estimates and unsupported-object blocks, races, save/reopen, group sizing/dimensions and real share downloads. Mocked analysis tests demonstrate UI contracts, not recognition accuracy.

Natural group fixtures and visual pose/garment review evidence remain in ignored `.local/group-validation-v3`. They include distinct people and documented failure cases; no broad accuracy benchmark is claimed. Never treat matching person counts alone as proof of correct poses or clothing.

Release validation includes Python geometry/API/updater tests, browser workflows and a production build. A real four-person hiking photo produced four figures with long sleeves/trousers and individual 3 mm bases in a watertight STL (103,598 triangles; 206.736 × 47.160 × 92.300 mm). All four base-stability heuristics passed; exact STL reimport matched API dimensions within 0.01 mm. Visible body-joint projection was preserved numerically. This does not establish that hidden joints or recognition are always correct. Cameras/backpacks in this photo remain outside supported detailed reconstruction. See the [actual four-person model](examples/group-photo-four.png) and [modeled garment/accessory detail example](examples/detail-preview.png); the latter is a synthetic style demonstration, not a recognition result.

## API

FastAPI `/docs` and `/openapi.json` document typed contracts. Application version is 3.1.0; project format stays `version:1` with backward-compatible defaults.

- `GET /api/health`: version, pose/group/clothing availability, six-person cap, mm/Z-up.
- `POST /api/analyze`: multipart `file`, optional `regions` JSON containing up to six normalized rectangles `[left,top,right,bottom]`. Returns `subjects`, `detected_people`, `unresolved_regions`, `overflow`, detection status and uncertainty notes.
- `POST /api/project/check`: normalize/save a project, resolving uncertain photo garments and retaining bounded estimation notes/evidence, including an empty draft.
- `POST /api/jobs`: generate the complete scene; poll `GET /api/jobs/{id}`.
- `POST /api/validate`: per-person, shared-base and combined mesh reports/bounds.
- `POST /api/export/stl`: `{ "project": Project }` exports the full scene and verifies exact binary STL reimport.

New photo projects include `photo_review` (detected count, unresolved count and confirmation). Jobs/validation/export reject empty, unconfirmed, unresolved, overflow, omitted-person and major unsupported-object scenes. All geometry endpoints apply the same automatic garment estimates, including older saved projects. Legacy integrations retain bounded preset parameters for old projects; these controls are absent from the new photo UI.

This remains a local single-user app. Public multi-tenant deployment needs authentication, isolation, scheduling, storage and operational work.
