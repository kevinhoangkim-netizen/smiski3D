# Architecture

## Pipeline and state

React/TypeScript project → bounded version-1 Pydantic schema → background mesh job → pose-fitted original implicit body → closed marching-cubes surface → fused clothing/accessories → exact figure height → world transforms → optional inscribed shared pedestal → validation and binary STL.

App state stores up to six figures, occasion/profile settings, individual style/pose/transforms and shared-base settings. Browser drafts/JSON projects contain parameters and landmark coordinates, not photos. The frontend normalizes old projects with new defaults. Undo/redo records project edits. Request epochs, scene revisions and AbortControllers prevent late analysis/import/generation results from replacing a newer scene. Validation reports retain the exact submitted signature and remain stale after later edits.

## Photo analysis

`pose.py` runs MediaPipe Tasks IMAGE mode on the CPU for up to six people. A full-frame pass plus overlapping windows can recover smaller subjects. Candidate coordinates map back to the original image; head/pelvis matching suppresses duplicates. This is approximate detection, not a universal people counter. Blur, cropped feet, occlusion and small subjects produce advisories.

Each subject preserves normalized landmarks plus `photo_aspect` (width/height). Geometry multiplies normalized X/depth by that aspect before evaluating directions, keeping equal physical pixel displacements consistent across portrait/landscape photos. Torso direction, shoulder line, bent limb directions and head tilt affect the sculpt. Canonical stylized limb lengths and conservative depth avoid claiming accurate unseen anatomy. Relative-height/layout suggestions use projected anatomical lengths and pelvis positions, with explicit perspective warnings. Original physical height and floor depth cannot be recovered from one view.

`appearance.py` runs quantized CLIP vision/text ONNX models using two CPU inference threads, without PyTorch, a GPU or paid APIs. Person/head/hand crops are compared with supported outfit/accessory descriptions, including negative/unsupported alternatives. Decisions require minimum cosine similarity and separation from alternatives. Raw cosine/margin is not displayed as a probability. Uncertain fields are left unset and flagged; corrections remain available. This cannot reconstruct arbitrary clothing or guarantee tiny/occluded accessory recognition.

The installer pins an immutable Hugging Face repository revision, verifies LFS SHA-256 values and the tokenizer Git object ID, then stores SHA-256 values in a local manifest. Inference verifies those files before opening the runtime. Downloads never contain user images, and inference never fetches weights implicitly.

## Original geometry and print conventions

The body has a large rounded head, almost no visible neck, shallow recessed dot eyes/mouth, rounded belly, uniform capsule-chain arms/legs and small feet. There is no protruding nose or imported official sculpt. A smooth signed-distance union sampled at 0.55 template-mm is reconstructed with marching cubes, then simplified by 0.025 template-mm where safe. Accessories are real solids fused with Manifold, not display-only props or overlapping shells. Bouquets have supported flower clusters/stems; veils are thick stylized forms rather than cloth simulation.

Body and STL use right-handed **Z-up millimeters**, front approximately negative Y. A figure is normalized to `height_mm` after individual base/accessory fusion. Uniform scale, Z rotation and XY translation follow. A shared base suppresses individual bases for the combined scene and lifts figures slightly into its surface. Separate figure export retains that figure's configured individual pedestal. Shared base margin/thickness are real world mm; individual base dimensions scale with the figure. Inscription is raised, portable glyph geometry fused to the base; supported text is bounded ASCII names/dates/punctuation.

`project_scene` returns world-coordinate meshes and reports, including a reserved `__group_base__` node that users cannot choose as a figure ID. API render data inversely removes each figure's placement transform; Three.js reapplies the same transform. The special base stays in world coordinates. Optional per-face material labels darken only the recessed facial marks in the preview/GLB; exported STL retains the real solid surface without color. Rename edits synchronize export mesh names immediately.

Group exports perform an actual union. Disjoint figures can remain separate closed solids; an encompassing shared base produces a connected assembly. A small Manifold tolerance handles numerical intersection slivers; final vertices are quantized/welded at STL float32 precision, and only zero-area collapsed triangles are removed. Remaining invalid topology is rejected, not disguised by filling holes. Export reimports the exact binary STL and verifies it again.

## Checks and measurements

Checks evaluate closed edge incidence, winding, positive signed volume, degenerate triangles and connected components on actual geometry. Parameter-derived structural feature estimates include relevant accessory stems/brims but do not solve local wall thickness. Stability compares mass projection to the planar contact hull/contact area. Supports, every overhang, material shrinkage, full vertex topology and drain holes are not solved. Checks never guarantee a successful print.

Bounds use transformed vertices; two-point measurement is world-space Euclidean distance. GLB converts mm/Z-up to meters/Y-up. STL has no units, so users must select millimeters in a slicer. No automatic fit-to-build-volume scaling occurs.

## Sharing and privacy

PNG captures the 3D canvas, not the reference-photo DOM. WebM records an actual eight-second orbit using MediaRecorder, with feature detection and a user-visible fallback. GLB carries visible meshes/materials. The downloaded HTML preview contains only figure names, dimensions, indexed geometry and an embedded WebGL viewer. It escapes JSON/script delimiters, uses textContent for labels, blocks network connections through CSP, and needs no external libraries/CDN. It supports rotate/zoom, keyboard controls, fit, spin and image export. No public link, photo gallery or hosting service is created.

Uploads are limited before multipart parsing to roughly 10 MB including envelope overhead. Uploaded files may briefly spool to framework-managed OS temporary files; the handler closes/deletes that spool immediately after its bounded read. JPEG/PNG/WEBP decode has a 20 MP guard, EXIF orientation and a 1600-px thumbnail limit. Application code does not persist photos. Images can remain in request/inference memory until an active analysis finishes, even if the browser cancels. Removing a reference revokes its browser object URL; deleting a draft removes browser-saved project parameters. These controls do not claim secure erasure of operating-system memory.

JSON contains no code definitions or arbitrary file reads. Sizes, names, identifiers, counts and supported text are bounded. Geometry concurrency and pending/result counts are capped. Results are in memory, not durable; server restart discards jobs. The backend is a local single-user application without public multi-tenant authentication/isolation.

## Local Windows launch

The distributed ZIP includes frontend production assets. `Start Little Figures.bat` locates Python 3.12 or opens the official installer. The standard-library launcher creates a private venv, installs pinned dependencies, downloads the pose/appearance models, starts a loopback-only FastAPI process, waits for app-specific health and opens `/welcome/`. Static frontend routes mount after API routes. No Node.js is needed for this prepared ZIP. Native Windows execution is not verified by the Linux test environment; native wheel availability was checked.
