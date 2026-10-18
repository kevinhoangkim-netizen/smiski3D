# Current limits and validation scope

## Available baseline

Manual-pose generation produces real original solid meshes with a browser editor and binary STL export. All six presets are deterministic. Individual and group layouts retain explicit millimeter scales. JSON projects reopen parameters; GLB is an optional color preview.

## Photo inference: installed and exercised, limited coverage

The initial official-model download returned HTTP CONNECT 403. After the required `storage.googleapis.com` network setting was saved, a subsequent download succeeded with normal TLS verification. The model is installed in ignored `.local/pose_landmarker_lite.task`. This environment needs no paid API or credential.

Real MediaPipe CPU inference was exercised on the official public `mediapipe-assets/pose.jpg` sample (one detected person, mean joint visibility 0.994) and a side-by-side two-copy fixture (two detected people). Both produced actual pose-driven STL geometry, reimported as watertight and consistently wound, at 90.000 mm total height. The tiled fixture checks multi-person plumbing; it is not evidence for diverse group photos or crowded-scene accuracy. No broad accuracy benchmark, genuine distinct-person group set, or three-person real-photo regression was run.

The browser workflow also verifies real two-person upload, subject exclusion and STL download when those local fixtures and the model are available. It is explicitly optional/skipped on a clean checkout without the reference fixture/model. Mocked failure tests verify error contracts, not estimator accuracy. Photos are not fabricated and manual output is not called inferred.

## Modeling

- Approximate original collectibles, not scans, identity reconstructions or accurate unseen surfaces.
- Canonical anatomy, clamped inferred depth, and neutral low-confidence limbs. Complex occlusion, crossed limbs, lying poses and severe perspective can produce poor fits even after the estimator is enabled.
- Implicit primitives are sampled at 0.65 template-mm spacing, with 0.025 template-mm surface simplification; finite tessellation can be visible at close zoom. No hair/clothes reconstruction, detailed fingers, hollowing, resin drain holes, supports or toolpaths.
- Selecting overlapping figures can require the subject list. Moves are constrained to the build plane and rotations to the vertical axis; height/scale are uniform. Camera presets are axis-aligned perspective views, not orthographic cameras.
- Pedestal margin/thickness are canonical-template dimensions scaled with target height, not independently fixed final physical dimensions. The UI states this. Dimension cards display actual mesh bounds after generation.
- Undo/redo applies to project edits, not camera positions or photo uploads. No autosave. Save a JSON project before closing the tab.

## Printability

Watertightness, edge manifold incidence, winding, components and degenerate triangles are direct geometric checks. They do not detect all vertex topology defects, self-intersections introduced by arbitrary external meshes (not accepted by the app), or printer-specific faults. Structural thickness is a parameter estimate, not a full local thickness solver. Stability uses contact-area/hull and center-of-mass heuristics. Raised arms and seated/leaning postures may require supports even when checks pass. No claim of guaranteed printability is made.

STL stores no units. Import as millimeters. Group exports can contain multiple disconnected solids by design; sliced arrangement and printer build volume must be checked independently. GLB uses meters/Y-up and should not replace the printable STL.

## Runtime

Tested on this Linux CPU environment with Python 3.12.14, Node 24.19.0 and system Chromium. Other OS/Python/browser combinations are not verified. Dependencies/model require Internet access to install; use the prepared environment snapshot afterward. WebGL requires browser support. The local API has no user authentication and should not be exposed as a public production service.

## Most useful next steps

1. Expand real-input regression coverage to a consented set of full-body, seated, low-confidence and distinct multi-person photos, including three people and difficult occlusion.
2. Refine anatomical fitting with world landmarks, ergonomic constraints and per-joint controls, then add pose-specific regression photos.
3. Add a full local thickness/overhang analysis and more robust support/stability feedback for actual printer/material profiles.

## Observed checks

- 32 Python API/geometry tests passed, including all six manual poses, numeric measurements, defect detection, upload bounds, stable job/export flow and STL reimport.
- TypeScript checking and the Vite production build passed.
- All 4 Chromium tests passed here, including real two-person upload/exclusion. The suite exercises the real WebGL editor, exact-height STL download, mobile layout/error handling, group layout/exports, transforms/undo, project reopening and meter-converted GLB.
- Separate real-inference smoke commands passed for the official one-person sample and two-copy fixture. CPU MediaPipe emits upstream TFLite feedback/projection warnings; inference completed despite them.
- The repeatable install script was exercised after dependencies were installed. No package-signature, checksum or TLS verification was bypassed.

## Windows local package

The prepared ZIP includes frontend production assets and a double-click Windows batch launcher. It requires Python 3.12 and first-run Internet access for Python dependencies and the optional pose model; Node.js is unnecessary for the prepared ZIP. The Python bootstrap and same-process static/API server were validated on Linux. The Windows-specific batch entry and Windows dependency wheels were not executed on a native Windows machine.
