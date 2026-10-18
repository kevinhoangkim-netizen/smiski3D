# Little Figures Studio 2

Make an original rounded figurine scene from a photo or manual poses, then edit, inspect and export actual printable solid geometry. Runs locally without an account or paid API. The supplied visual references informed the large round head, tiny recessed face, soft body, continuous limbs, and small feet. No official sculpt or character mesh is included, and no affiliation is claimed.

**[Download the Windows app](https://github.com/kevinhoangkim-netizen/smiski3D/raw/refs/heads/main/downloads/Little-Figures-Studio-Windows.zip?version=2)**

## Windows: no development commands needed

1. Extract the entire ZIP first.
2. Double-click **Start Little Figures.bat**.
3. If Python 3.12 is missing, the official installer download opens. Install it, then open the launcher again.
4. Keep the launcher window open while first-time setup downloads dependencies and the two local vision models. The browser opens when the app is ready.

The ZIP includes the built website, so Node.js is unnecessary. The first setup needs Internet access; later launches reuse downloads. The outfit model is about 149 MiB; it runs on the CPU without PyTorch or a GPU. Model downloads use HTTPS and the outfit installer verifies upstream checksums at a pinned revision. No photo is sent to a model provider.

For an upgrade, save any project you want to keep, close the old launcher window, and extract the new ZIP into its own folder. Existing version-1 project JSON files reopen with defaults for new fields. A previously saved photo pose without an image aspect ratio uses the legacy square default; reanalyze the original photo to recover correct aspect-aware directions.

The launcher/server were tested on Linux. Windows Python 3.12 binary wheels exist for the native dependencies, but a native Windows session has not been exercised here. Send the launcher error text if setup fails. See `START HERE - Windows.txt`.

## What the studio does

- **Up to six people:** local MediaPipe inference with overlapping-window passes for some smaller group members, duplicate suppression, crop/blur/occlusion advisories, approximate placement and relative-height suggestions.
- **Preserved gestures:** photo torso lean, shoulder direction, head tilt and bent limbs influence the sculpt. Width/height normalization is corrected for landscape and portrait photos. Hidden limbs and single-view depth remain uncertain.
- **Automatic styling suggestions:** local quantized CLIP compares each person's clothing/head/hand crops with supported outfit/accessory descriptions. Supported shapes are casual clothing, suit, wedding dress, cap/top hat/wide-brim hat, bouquet, veil and bow tie. Uncertain fields remain unset for review. This is closed-set similarity classification, not a general reconstruction of any garment.
- **Per-person corrections:** rename/select a person, change pose or outfit, choose accessories and bouquet hand, and include/exclude people without rebuilding the whole project from scratch.
- **Real geometry:** rounded implicit surfaces, fused clothing/accessory parts, real recessed eyes/mouth, and flat optional pedestals. Head-to-body proportions and limbs follow the supplied minimalist reference direction. Dark facial material is a preview/GLB treatment; STL contains physical recesses and no colors.
- **Group keepsakes:** round or rectangular shared pedestal, raised names/dates/inscription, and occasion presets. Pedestals fuse touching figures into one scene; separate figures can also export individually.
- **Useful editing:** orbit/pan/zoom, group-fitting camera, planar move and vertical rotation gizmos, numeric transforms, uniform scale, overall height, undo/redo, millimeter bounds and two-point surface distance.
- **Save/recover:** explicit JSON save/open and automatic browser-local draft recovery. Remove the reference photo independently; delete the saved draft from the browser. Project files and drafts include pose coordinates/parameters, never image bytes.
- **Share locally:** PNG image, actual 8-second WebM turntable, GLB, and a self-contained interactive HTML preview. Send the HTML file to someone to open locally: it needs no app account, CDN or server and contains no source photo. It does contain figure names and geometry. No hosted public URL/gallery is created.

Photo-derived figures are stylized approximations, not likeness scans. Review hands, clothing and interactions before printing. Touching/occluded people remain harder than separate full-body subjects. All edits can be saved and corrected.

## Printing

Generate after geometry changes. Transform edits update the view immediately; when a shared base is enabled, regenerate it after moving people. A pending-changes notice prevents stale geometry from being exported.

Target height includes an individual pedestal and resets the frontend uniform scale. Pedestal margin/thickness are template dimensions that scale with the figure. Shared pedestal dimensions are actual group-world mm. Group scenes retain each person's scale; no silent fit-to-printer deformation occurs.

Run checks for watertightness, edge incidence, winding, degenerate faces, components, approximate structural feature size and contact/stability. Export refuses topology that does not survive actual STL reimport. These checks do not guarantee a print: inspect supports, orientation, drainage, actual wall thickness and build volume in your slicer. Resin/FDM thresholds are editable heuristics.

**STL stores no units. Import it as millimeters.** GLB previews convert to glTF's meters/Y-up convention and preserve color. Straight-line measurement is not a geodesic surface measurement.

## Development setup

Tested baseline: Linux CPU, Python 3.12.14, Node 24.19.0, npm 11, Chromium. Other environments are not fully verified.

```bash
bash scripts/install.sh
bash scripts/dev.sh
```

The development frontend uses port 5173 and proxies `/api` to FastAPI on 127.0.0.1:8000. Optional `LITTLE_FIGURES_FRONTEND_PORT` and `LITTLE_FIGURES_BACKEND_PORT` change development/test ports. `CHROMIUM_PATH` can select another browser executable for tests.

The same-process local production launcher serves the compiled studio and `/welcome/` landing page with FastAPI on an available loopback port, starting at 8765. It opens the browser after health checks. Keep the launcher open; Ctrl+C stops its child server. It does not expose a public listener or require a hosting subscription.

No secrets are required. `.env.example` lists optional model paths. Export variables in your shell; `.env` is not automatically loaded by the backend.

### Vision model installation

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python -m pip install -r backend/requirements-appearance.txt
.venv/bin/python scripts/download_appearance_model.py
```

The Windows launcher and installation script perform these steps automatically. The small pose model downloads from `storage.googleapis.com`. The pinned outfit model downloads from `huggingface.co` and its actual download CDN `us.aws.cdn.hf.co`. The downloader verifies the model's upstream SHA-256 and tokenizer Git object checksum before writing a local manifest; inference verifies the installed SHA-256 values before use. Missing/unusable model files are reported explicitly rather than converted into fake photo results. Models are not redistributed inside the source ZIP.

## Validation and examples

```bash
.venv/bin/python -m pytest -q
npm run build --prefix frontend
npm run test:e2e --prefix frontend
.venv/bin/python -m scripts.generate_examples
.venv/bin/python -m scripts.verify_photo /path/to/full-body-photo.jpg --expected-people 2
.venv/bin/python scripts/package_windows.py
```

Python tests cover geometry/export, every manual pose and supported accessory, aspect-corrected photo gestures, shared inscribed bases, exact dimensions, defect detection, inclusion/exclusion, schema migration, bounded uploads/queues, and inference abstention/error contracts. Browser tests exercise WebGL, editing, recovery, comparison, cancellation, stale-response protection, real binary downloads, GLB units, actual video and offline preview rendering/privacy. Mocked analysis tests validate the UI contract, not model accuracy.

Real-input fixtures are kept only in ignored `.local/`; they are not redistributed as repository photos. The official public MediaPipe pose sample and tiled copies check one/two/six-person plumbing, not broad multi-person accuracy. See [LIMITATIONS.md](LIMITATIONS.md) for the precise verification scope and known failure modes.

Release validation: 98 Python tests passed, the production build passed, and all 16 browser scenarios passed across the full run and focused reruns. A real wedding photo also completed the production launcher's analyze/generate/validate/export flow: two people and an inscribed shared pedestal became one watertight STL solid, with reimported dimensions matching validation within 0.01 mm. Native Windows remains untested.

[Example STL](examples/standing-100mm.stl) · [Example project](examples/standing.project.json) · [Architecture](ARCHITECTURE.md)

## API

FastAPI `/docs` provides interactive OpenAPI; `/openapi.json` contains the typed contracts. Unknown fields and unsupported accessory values are rejected. Project `version:1` remains backward compatible; application version is 2.0.0.

| Method / path | Input | Result |
| --- | --- | --- |
| GET `/api/health` | — | App/version, pose/appearance availability, max people, mm/Z-up |
| GET `/api/poses` | — | Supported manual presets |
| POST `/api/analyze` | Multipart image `file` | People, landmarks/aspect, approximate layout/heights, styling suggestions and uncertainty |
| POST `/api/project/check` | Project JSON | Normalized validated project; old projects receive new defaults |
| POST `/api/jobs` | Project JSON | Stable job ID, 202; bounded queue |
| GET `/api/jobs/{id}` | Job ID | queued/generating/validating/ready/error; ready includes indexed meshes and reports |
| POST `/api/validate` | Project JSON | Per-object and combined checks/bounds |
| POST `/api/export/stl` | `{ "project": Project, "figure_id": null }` | Binary combined STL; use included ID for a separate figure |

At most two geometry operations run concurrently, four pending jobs are accepted, and at most twelve in-memory results are retained. Completed results expire after 30 minutes when new work arrives. Photo inference is serialized and no photos are stored in jobs. This is a local single-user app; public multi-tenant hosting still needs authentication, isolation, stronger request scheduling and operational controls.

## Troubleshooting

- **Wrong/old app opens:** close the earlier launcher window and restart the new extracted package. A different occupied port causes the launcher to choose another local port.
- **First setup is slow:** dependencies and model weights are downloaded once. Keep the launcher open and connected to the Internet.
- **Outfits unavailable:** inspect the launcher error; allow the model host/CDN if your network blocks it, then reopen the launcher to retry. Manual styling and pose workflows remain usable.
- **No people or wrong count:** choose a clear full-body photo. Crowding, hidden faces, small subjects and overlapping people can cause misses/duplicates. Review subject checkboxes before generating.
- **Uncertain/wrong outfit:** use per-person correction controls. The supported category set is limited and small accessories need enough visible pixels.
- **Cannot reach backend in development:** start both services with `scripts/dev.sh` and check that frontend/backend port overrides agree.
- **Unwritable npm cache:** use the install script, which selects `.local/npm-cache`.
- **WebGL/video unavailable:** enable browser hardware acceleration or use another supported browser. PNG/offline-preview alternatives are available when WebM recording is unsupported.
- **Sharing:** send the downloaded HTML file, PNG or WebM. These are file exports; they do not publish the photo or create a website URL.
