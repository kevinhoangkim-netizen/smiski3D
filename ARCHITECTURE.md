# Architecture

## Data flow

React/TypeScript controls → bounded version-1 Pydantic project → background mesh job → original procedural skeleton → rounded implicit primitives → smooth signed-distance union → marching-cubes surface → Manifold solid union → integrated recessed face marks → fused optional pedestal → exact millimeter height → layout transforms → checks and binary STL. Three.js renders indexed geometry returned by the API; no backend image rendering is used.

`frontend/src/App.tsx` owns the project, capped undo/redo snapshots and asynchronous status/error states. `Viewport.tsx` owns WebGL resources, orbit controls, planar move/vertical rotation gizmos, raycast selection and surface-point measurement. Mesh parameter changes require generation; world transforms update the preview immediately. Print reports are marked stale after project changes. Saved JSON contains the authoritative parameters/landmarks, not bulky generated meshes or photos.

The geometry code caches up to 48 canonical parameter sets in memory and copies each mesh before applying transforms. Render geometry is returned without layout transforms; React applies the same uniform scale, Z rotation and XY translation as the backend export path. Selection and controls remain separate from the exported mesh. GLB exports convert to the glTF meter/Y-up convention.

## Original geometry

No external sculpt or figurine mesh is used. Torso, hips, head, chunky feet/hands, neck and rounded limb segments are defined procedurally. Anatomical landmarks set skeleton direction. Capsule limbs intersect the torso and adjacent joints. A smooth signed-distance union is sampled on a 0.65 template-mm grid and reconstructed with marching cubes, producing one soft closed surface. Manifold simplification removes redundant triangles; the pedestal is fused with a true boolean union. Eye marks are shallow subtractions and the nose is fused into the head. Pedestals overlap the feet before union. Six deterministic templates provide standing, seated, reaching, leaning, crouching and arms-raised poses.

Group exports union all requested solids, including pedestals. Disjoint objects remain separate closed solids; touching objects fuse. A 0.0001 mm Manifold tolerance/simplification reduces intersection slivers. Final geometry is quantized to STL float32 precision, welded, and only triangles collapsed by that quantization are removed before topology is checked. No holes are filled to force a passing result. Export checks reject open or inconsistently wound results instead of advertising an invalid mesh as printable.

## Coordinates and dimensions

The backend, editor scene, numeric fields and STL use **right-handed Z-up millimeters**. Front faces approximately toward negative Y. XY is the build plane, Z=0 its surface. Each figure is normalized to `height_mm` after its base is fused. `transform.scale` is applied after height normalization, followed by vertical-axis rotation and XY translation. No automatic layout scaling occurs. Photo subject bounding-box centers set an approximate horizontal group arrangement; users can edit it explicitly. A target-height edit resets frontend uniform scale to 1 so the result matches the requested total height. Base margin/thickness are defined in canonical template mm, then scale with the complete figure; this is stated in the UI.

STL stores no units. Users must choose millimeters in their slicer. The GLB exporter wraps the scene in a -90° X rotation and 0.001 uniform scale so viewers receive meters/Y-up. Two-point measurement is straight-line distance between world-space raycast surface points, not a surface/geodesic path. Dimension cards use transformed mesh vertices rather than approximate screen pixels.

## Pose analysis and limitations

`backend/app/pose.py` isolates the replaceable estimator. MediaPipe Tasks Vision supports `num_poses=3`, IMAGE mode, CPU execution and visibility scores. The official model is downloaded separately; inference needs no API key. An initial cloud policy denial was resolved by adding the official model host; the model is installed here. The real-photo path was verified on the official public single-person sample and a tiled two-copy fixture, including pose-driven STL reimport. The manual pipeline remains available without the model. This does not establish broad detection or fitting accuracy.

When available, the estimator receives an EXIF-oriented RGB thumbnail in process. Image-space shoulder handedness sets left/right attachment positions; image landmarks set limb directions while canonical torso width, limb lengths and head proportions preserve the original stylization. Conservative depth uses a clamped portion of estimated depth. Visibility below 0.5 falls back to neutral limb positions rather than asserting knowledge of hidden body parts. Low overall confidence produces warnings. Facial identity, clothing, hair and unseen surfaces are not reconstructed. Severe overlap, crossings, lying poses and foreshortening remain difficult; a manual override is always available.

## Checks

Closed edge incidence, consistent winding, signed positive volume, near-zero triangle area and connected components are evaluated on actual generated mesh geometry. Stability compares center-of-mass projection with the convex hull of coplanar build-plate contact, plus flat contact area. This is approximate and does not simulate forces/materials. A parameter-derived minimum structural limb diameter is compared to the editable resin/FDM threshold; it does not measure every local wall, recess, joint, overhang or printer exposure effect. Warnings are retained, not hidden by repairs. These checks never guarantee print success, produce supports or generate toolpaths.

## Privacy and execution boundary

The browser sends a selected photo only to the local FastAPI instance. Bytes are bounded at 10 MB, decoded as JPEG/PNG/WEBP with a 20 MP guard, then reduced for inference. No filename is used for filesystem paths. The raw request stream is bounded before multipart parsing. The framework may spool uploads larger than 1 MB to an OS temporary file; the handler closes the upload immediately after its bounded read, deleting that temporary file. Browser object URLs and in-memory decoded images persist only for the active browser reference/request. No application image files, persistent image job payloads, third-party uploads or analytics are created. The request releases its decoded image when inference ends. Project JSON contains pose coordinates and parameters, which users should still treat as personal data if derived from private photos.

Jobs use random stable IDs, an in-memory bounded queue and two work slots. Results are not durable; process restart discards them. Expired completed results are cleared when new jobs arrive. Request validation rejects arbitrary code, path strings and unknown fields. The app accepts no filesystem mesh imports. This is a single-user local MVP without authentication or production-rate controls. Multi-user deployment would require isolation, authentication, stronger quotas, timeouts, and durable job storage.

## Local Windows launch and landing page

`frontend/public/welcome/` is a static landing page copied into the Vite production build. Its studio links open the real root app on the same origin. Artwork is rendered from the application's own procedural meshes. FastAPI mounts the built frontend after all API routes, allowing one loopback-only process to serve both the landing page/studio and the geometry/photo API.

`Start Little Figures.bat` locates Python 3.12 on Windows, or opens the official installer download when missing. The standard-library `scripts/launch_local.py` creates a private virtual environment, installs the pinned lockfile, fetches the optional official model, checks a stable app-specific health identity, starts Uvicorn on an available local port, and opens the browser after readiness. It terminates its child process on exit. The distributed ZIP includes built assets, avoiding a Node.js prerequisite. Native Windows behavior remains unverified in this Linux environment.
