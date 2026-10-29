# Start here: Little Figures Studio AI handoff

This archive contains the editable app, tests, dependency locks, launchers and
example output images. Application version: **3.1.0**. The application baseline
is Git commit `763842593a76971b1f3f01f6c0fbbd19735bc712` in
<https://github.com/kevinhoangkim-netizen/smiski3D>.

## What the owner wants

The owner is not a coder and uses Windows. They want to upload a group photo
and automatically create one printable 3D scene of **everyone** in the picture,
keeping their natural poses, arrangement, visible clothing and accessories.
The desired look is a Smiski-like rounded pale-green collectible: a large round
head, tiny eyes and mouth, a small rounded nose, smooth compact body and limbs.
The current sculpt is procedural and original; no official character mesh is
included. The reference photographs previously discussed in chat are not in this
archive; do not assume access to them.

The main improvement priorities are:

1. Better photo pose accuracy, especially bent/crossed arms, held objects,
   sitting, hugging, overlap, and which limbs belong to which person.
2. Better clothing and accessory recognition **and** geometry. The owner finds
   the current quality too crude. Preserve suits, dresses, hats and bouquets
   from the actual photo instead of substituting a generic stock costume.
3. More convincing rounded character proportions and facial detail, including
   noses. The latest version has noses but still needs visual review.
4. Stable printable bases: individual round/rectangular bases or a shared
   pedestal, with reliable attachment and dimensions in millimeters.
5. A simple local Windows experience with incremental updates, saved projects
   and useful preview/export, without repeated full reinstalls.

Keep pose, outfit and accessory selection automatic. The owner explicitly
asked for those manual menus to be removed. Unclear clothing should be inferred
sensibly and labeled as estimated, rather than blocking the entire group.
Do not silently omit people. The current implementation supports up to six;
unresolved people and unsupported major objects remain explicit problems.

## Current implementation and limits

This is **not a general photo-to-3D reconstruction model**. It combines local
YOLOX person detection, MediaPipe pose estimation, SCHP clothing segmentation
and CLIP similarity cues with procedural figurine geometry. Increasing polygon
counts alone will not fix incorrect detections, garment labels or hidden poses.
Inspect both the vision pipeline and the sculpting logic.

Version 3.1 added bounded garment evidence and automatic estimates; physical
hems, cuffs, lapels, folds and accessory details; small noses; physical-thickness
bases; and visible joint mapping that preserves image-plane proportions instead
of assigning every limb segment a fixed length. Hidden joints and single-view
depth remain uncertain. Hand contact and group interaction can still change,
especially when enlarged heads require more spacing or subjects are grounded
independently. Clothing classification and accessory ownership can be wrong.

Generation, validation and STL export must use the same normalized project.
Keep estimation provenance visible and preserve old saved projects. Avoid
claiming that a guessed garment or a mocked browser response demonstrates
accurate recognition.

The app runs locally without paid APIs, a GPU or an account. Preserve that
default. Explain any proposed change to those requirements. Photos stay local;
project/share files exclude original photo bytes. Keep downloaded models,
the private Python environment and saved projects during app updates.

## Where to work

| Area | Files |
| --- | --- |
| People detection and matching | `backend/app/people.py`, `pose.py` |
| Clothing/accessory recognition | `appearance.py`, `garment_parser.py` |
| Uncertain clothing estimates | `garment_estimation.py` |
| Body, face, joint mapping, print checks | `geometry.py` |
| Garment and accessory solids | `figure_style.py` |
| Shared bases and lettering | `pedestal.py` |
| Project contracts and API | `schemas.py`, `main.py` |
| Photo/group workflow | `frontend/src/App.tsx`, `PhotoReview.tsx`, `types.ts` |
| Base controls and sharing controls | `frontend/src/StudioFeatures.tsx` |
| 3D preview and exports | `Viewport.tsx`, `sharing.ts` |
| Windows launch/update | `scripts/launch_local.py`, `update_local.py`, root `.bat` files |
| Backend/mesh/update tests | `tests/` |
| Browser workflow/race/export tests | `frontend/tests/` |

Backend filenames in the table are under `backend/app/`; frontend filenames
are under `frontend/src/`. Read `ARCHITECTURE.md`, `README.md` and
`LIMITATIONS.md` before changing the pipeline. Some older documentation still
describes previous pose behavior; the current code and focused regression tests
are authoritative for the implemented mapping.

## Set up and verify

Use **Python 3.12** and **Node.js 24**. On Linux, from the extracted project:

```bash
bash scripts/install.sh
bash scripts/dev.sh
```

The installer uses locked dependencies, downloads about 255 MiB of local vision
weights, builds the UI and runs Python tests. Model downloads require internet;
inference uses local files. Do not bypass TLS or model checksum verification.
SCHP preprocessing requires BGR and aspect-preserving black padding; switching
to RGB or direct square resize previously produced wrong garment results.

Useful checks after changes:

```bash
.venv/bin/python -m pytest -q
npm run build --prefix frontend
npm run test:e2e --prefix frontend
.venv/bin/python -m scripts.verify_photo /path/to/group.jpg --expected-people 4 --shared-base
```

Browser tests require Chromium; set `CHROMIUM_PATH` if necessary. API/Vite ports
can be overridden together using `LITTLE_FIGURES_BACKEND_PORT` and
`LITTLE_FIGURES_FRONTEND_PORT`. Avoid simultaneous Playwright runs writing to
the same output directory.

The prior release reported **255 Python tests and 28 browser scenarios passing**
on Linux CPU/Chromium, plus a successful production build. This is historical
evidence, not a new test run in your environment. Native Windows execution was
not tested. Browser analysis fixtures are partly mocked, and private real-photo
fixtures/evidence are excluded here. Obtain suitable photos and visually verify
count, limb ownership, poses, clothing and accessories; also reimport actual
STL exports and check topology, dimensions, attachment and stability.

`examples/detail-preview.png` is a **synthetic style demonstration**, not a
recognition result. `examples/group-photo-four.png` shows the generated result
from a four-person photo, without the source photo. Other screenshots and manual
JSON/STL examples may predate 3.1; they are not accuracy benchmarks.

## Deliver improvements to the owner

Make focused changes and explain what improved and what still fails. Provide
a downloadable Windows package so the owner does not need to code. Build the
frontend before packaging:

```bash
npm run build --prefix frontend
python scripts/update_local.py --build-batch
python scripts/package_update.py
python scripts/package_windows.py
```

For a new release, update the version consistently in `app-version.json`,
`backend/app/main.py`, and frontend package/lock metadata. Use a new versioned
runtime ZIP instead of changing an already published release. The updater is
pinned to this GitHub repository's HTTPS release manifest; an unrelated fork
will need its own explicit release configuration. Do not publish to someone
else's repository without their authorization.

## Archive contents

`SOURCE_MANIFEST.json` records each included file's size and SHA-256, plus the
source baseline. The archive includes editable sources, pinned dependencies,
tests, documentation, scripts and example assets. It excludes `.git`, `.env`,
local runtime credentials, user photos/projects, downloaded models, installed
dependencies, compiled frontend output and existing release ZIPs. The tracked
`.env.example` contains comments and optional setting names, not secret values.
Build output and models can be recreated using the included scripts.
