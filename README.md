# Little Figures Studio

A locally runnable desktop-first browser studio for original rounded humanoid figurines. Choose a manual pose, edit a real 3D model, inspect dimensions and mesh checks, and download binary STL in millimeter coordinates. No account, paid API, analytics, or third-party photo upload.

**Working baseline:** six manual presets, up to three individually selectable figures, orbit/zoom/pan, move/rotate gizmos, numeric translation/rotation/scale, target height, undo/redo, pedestals, two-point measurement, mesh checks, individual/group STL, color GLB, and versioned JSON projects. A procedural demo loads immediately.

**Photo mode:** a real local MediaPipe multi-person integration is installed and verified here with the official public sample image and a tiled two-copy fixture. Both produced pose-driven watertight STL exports. This is a functional check, not an accuracy benchmark across different people/poses. Missing-model and no-detection cases remain explicitly labeled; manual poses always work without the model. See [LIMITATIONS.md](LIMITATIONS.md).

## Run on your Windows computer

**[Download the prepared Windows ZIP](https://github.com/kevinhoangkim-netizen/smiski3D/raw/refs/heads/main/downloads/Little-Figures-Studio-Windows.zip)**. If GitHub asks, sign in to the account that can access this repository.

The prepared Windows ZIP includes the built website, so **Node.js and coding commands are not needed**:

1. Extract the entire ZIP.
2. Double-click **Start Little Figures.bat**.
3. If Python 3.12 is missing, the launcher opens the official 64-bit Windows installer. Install it, then open the launcher again.
4. Let the first setup finish downloading dependencies and the official pose model. Your browser opens automatically to the landing page. Choose **Open the studio**.

Keep the launcher window open while using the app. Close it to stop the local server. Setup creates a private `.venv` and stores caches/model under `.local` in the extracted folder. Later launches reuse them. The server listens only on your computer's loopback interface; there is no public hosting or hosting bill.

The Windows batch entry point is included; the Python launcher and single-process production serving were exercised on the available Linux machine. A native Windows run has not been verified here. See `START HERE - Windows.txt` for the short version.

The landing page source is `frontend/public/welcome/`. It ships in the production build and opens the real studio on the same local server; there are no fake signups, testimonials, or payment links. The artwork renders actual original models from this app. Sites was unavailable in this chat, so no Sites publication was performed.

## Run

Recommended: Python **3.12** (tested 3.12.14), Node **24** (tested 24.19.0), npm 11, Linux CPU. Python 3.11+ and Node 22+ may also work, but other combinations have not been verified. No GPU needed. Python installation uses platform wheels for NumPy, SciPy, Manifold, OpenCV and MediaPipe. Unsupported platforms may require their native build toolchains; use the tested Linux environment first.

From the repository root:

```bash
bash scripts/install.sh
bash scripts/dev.sh
```

The development frontend listens on port **5173** and proxies `/api` to FastAPI on **127.0.0.1:8000**. Open port 5173 in your local browser. Keep the command running; Ctrl+C stops both services. The backend is bound to loopback. `.env.example` documents the one optional variable; export it in the shell, since `.env` is not auto-loaded.

Installation preserves the lockfiles, verifies normal TLS/package integrity, builds the frontend, and runs the backend tests. A failed optional model download is reported and leaves manual mode available. The first dependency install requires Internet access to PyPI and npm. A published cloud filesystem snapshot retains dependencies and the optional downloaded model, but services must restart for every new task.

To run separately, in two terminals from the repository root:

```bash
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
npm run dev --prefix frontend -- --strictPort
```

## Use

1. The default original figure is a **manual demo**, not an inferred photo result. Change a preset, head proportions or pedestal, then select **Generate figures**.
2. Add up to three figures. Checkboxes include/exclude them. Click a figure in the viewport or in the subject list to edit it. Moving and rotating affect the preview immediately. Numeric fields provide keyboard-accessible alternatives to gizmos. Undo/redo also support Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z outside text inputs.
3. Target height includes the pedestal and resets uniform scale to 100%. Pedestal thickness/margin are template millimeters and scale with the entire figure. The dimension readouts measure the current preview. Pending geometry changes must be generated before export.
4. The ruler tool selects two mesh surface points and displays Euclidean distance in mm. Front/side/top buttons use the perspective camera from axis-aligned positions; they are not orthographic projections. Reset camera restores the perspective view.
5. Run print checks after edits. Expand each row for its scope. Thickness and stability checks are explicitly heuristic, and overhang/support/drainage checks are not performed. Print profiles provide editable feature-warning defaults: resin 1.2 mm, FDM 2 mm.
6. Download an individual or group **binary STL**. STL has **no unit metadata**: import as **millimeters** in your slicer. A group export performs a true solid union where objects touch and preserves separated solids where they do not. Color GLB previews convert the internal Z-up millimeter frame to Y-up meters. GLB is a viewing format, not the printable export.
7. Save/reopen JSON projects to retain poses and parameters. Photos are never stored in them. Saving is explicit; closing an unsaved browser tab discards edits.

### Enable photo estimation

The official model is already installed in this prepared environment. For a fresh checkout, allow **storage.googleapis.com** in the environment's Internet settings if necessary, then run:

```bash
.venv/bin/python scripts/download_model.py
```

The script downloads the official versioned MediaPipe `pose_landmarker_lite` task over verified HTTPS into ignored `.local/`. The printed SHA256 is an informational fingerprint, not a claimed upstream checksum. No downloaded model is redistributed in this repository. Refresh the browser to update model availability.

Choose **From a photo** and upload JPEG/PNG/WEBP, at most 10 MB / 20 MP. Uploads may briefly spool to framework-managed temporary files, deleted on close immediately after the bounded read. Decoded images stay in memory, EXIF-oriented and downsampled to 1600 px for CPU inference. Up to three detected people are shown with confidence warnings. Detection is not an identity/likeness reconstruction. Hidden/uncertain limbs use neutral fallbacks; depth is conservative. Review the pose and generate explicitly. If no people are detected, no new models are fabricated and existing figures remain unchanged.

## Validation

```bash
.venv/bin/python -m pytest -q
npm run build --prefix frontend
npm run test:e2e --prefix frontend
.venv/bin/python -m scripts.generate_examples
```

The browser suite uses Playwright and system Chromium at `/usr/bin/chromium`, starts both services if necessary, tests a real rendered workflow and inspects the actual downloaded STL's binary triangles and height. It also checks transforms/undo, project reopening, GLB unit conversion, mobile overflow and invalid-image error handling. An optional fourth test exercises real two-person inference and exclusion when the local public fixture and model are installed; it is explicitly skipped if absent. For other systems, adjust `frontend/playwright.config.ts` to your Chromium path or install a Playwright-managed browser. Screenshots are generated in `examples/` and failure traces in ignored `.local/playwright-results/`.

The Python suite covers every manual pose, STL reimport, 100 mm target height, manifold/watertight solids, group inclusion and scale, overlapping solid union, measurement math, deliberately defective meshes, upload limits, schema validation, and explicit inference failure/no-detection handling. Mocked pose failures test contracts, **not** real photo accuracy. Real one-person and two-copy-photo inference was separately exercised end to end. See [LIMITATIONS.md](LIMITATIONS.md) for the observed results and scope.

To verify a real image through inference, generation, export and STL reimport:

```bash
.venv/bin/python -m scripts.verify_photo /path/to/full-body-photo.jpg --expected-people 1
```

The test reference used here is the official public `https://storage.googleapis.com/mediapipe-assets/pose.jpg` image, downloaded only to ignored `.local/pose-test.jpg`. A side-by-side copy is stored in ignored `.local/two-person-test.jpg` to test multi-person plumbing. These photos are not redistributed as repository assets. No new user photo is required to explore the manual demo.

[Example binary STL](examples/standing-100mm.stl) · [Example project](examples/standing.project.json) · [Manual pose data](examples/manual-poses.json)

## API

FastAPI's `/docs` provides interactive OpenAPI documentation; `/openapi.json` exposes the schemas. Unknown fields are rejected; names, figure counts, dimensions, landmarks and transformations are bounded by Pydantic. See `backend/app/schemas.py` for complete contracts.

| Method / path | Input | Result |
| --- | --- | --- |
| GET `/api/health` | — | Backend status, model availability, units and axis |
| GET `/api/poses` | — | Supported manual presets |
| POST `/api/analyze` | multipart image field `file` | Detected subjects/landmarks/bounds/warnings, or explicit 413/415/503 |
| POST `/api/project/check` | Version-1 Project JSON | Validated, normalized project; 422 for invalid input |
| POST `/api/jobs` | Project JSON | Stable job ID, 202; at most 3 figures; 429 when queue is full |
| GET `/api/jobs/{id}` | job ID | queued/generating/validating/ready/error; ready results include triangle geometry and reports |
| POST `/api/validate` | Project JSON | Per-figure mesh checks and combined bounds |
| POST `/api/export/stl` | `{ "project": Project, "figure_id": null }` | Combined binary STL; specify an included ID for one figure |

Jobs are in memory, bounded to 24 results with 30-minute expiration when new jobs arrive. At most two geometry operations run concurrently. Photos are not in jobs. This is a single-user local development application, not a hardened public multi-tenant service. Do not expose the development server as a public production API.

## Troubleshooting

- **Cannot reach backend:** run both services with `scripts/dev.sh`; the Vite `/api` proxy targets port 8000.
- **npm cache is unwritable:** the installer already uses `.local/npm-cache`. Use `npm ci --prefix frontend --cache "$PWD/.local/npm-cache"` for manual installs.
- **Model download returns 403:** allow `storage.googleapis.com` in cloud environment settings. Manual mode does not require it. No token is needed for the public official model.
- **Model runtime fails:** inspect the backend log, verify the Python lockfile installation and use a trusted supported `.task` model. Missing native system libraries can affect OpenCV/MediaPipe on other operating systems.
- **3D viewport unavailable:** enable browser hardware acceleration/WebGL. The UI reports this. Backend STL generation does not depend on WebGL.
- **No people detected:** try a clear full-body photo. Use an explicitly chosen manual preset if detection remains uncertain.
- **Print check warnings:** use a pedestal, inspect mesh orientation and supports in a slicer, and compare feature thresholds with your printer/material.

See [ARCHITECTURE.md](ARCHITECTURE.md) for coordinate and privacy contracts. The design follows the brief's high-level rounded collectible aesthetic; it uses no official sculpt, mesh, packaging, logo or character. No affiliation is claimed. Browser research tools were unavailable, so the supplied textual style description was used instead of external visual references.
