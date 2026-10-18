# Architecture

## Photo-first workflow

The React studio starts with an empty version-1 project. Uploading a photo clears previous generated data and submits its pixels to the loopback-only FastAPI backend. The new UI has no preset pose, outfit, accessory, occasion or manual placement controls. All detected people are represented in one project and one generation request. Selecting a person changes inspection only.

Analysis returns resolved subjects, independent detected-person count, unresolved regions, overflow and model status. Numbered outlines indicate people, not model crops. Generation requires user confirmation of the count. A recovery rectangle submits the same entire original photo plus additive normalized regions; it does not replace the group with a crop. Failed/overlapping recovery must remain unresolved unless an additional distinct identity is recovered.

Project.photo_review persists detected/unresolved counts and confirmation without pixels. API generation, validation and STL reject empty, unconfirmed, unresolved, overflow and omitted-person photo groups. Unidentified upper/lower garment categories and explicitly blocked unsupported objects also prevent generation. Legacy projects retain bounded old parameters and are labeled as saved preset scenes; those configuration menus are absent from the new workflow.

Request epochs, scene revisions and AbortControllers prevent late analysis/import/generation from replacing newer work. JSON/browser drafts contain names, landmarks, measured clothing fields and scene parameters, not photo bytes or filenames. Original images remain browser/request memory only, apart from short-lived multipart spooling that closes immediately after bounded read.

## Independent people and pose

people.py runs a pinned Apache-2.0 YOLOX-S quantized ONNX model on CPU. Global/full-height overlapping windows and optional retry regions produce independent body proposals. NMS and window-edge checks suppress duplicate slivers while retaining real smaller people.

pose.py runs MediaPipe on the original image and each full-height person region. Pose candidates are remapped to original coordinates, deduplicated by head/body evidence and matched to independently detected people. Crop-first selection reduces assigning one person's head to another person's body. Unusable poses remain unresolved; no detection box becomes a generic standing figure. Counts alone do not establish correct pose association, so real fixtures are also reviewed with landmark overlays.

Landmarks preserve image aspect (width/height) before geometry evaluates directions. Torso lean, shoulder slopes, head tilt and bent limb directions influence canonical rounded anatomy. Back-facing anatomical shoulder order reverses the face/garment-front surface without rotating/mirroring the photographed gesture. Relative height and XY placement are estimated from visible projected anatomy. A least-movement horizontal spacing pass keeps enlarged heads distinct, preserves person order/heights/limb directions and warns when spacing/contact changes; it does not infer extra floor depth. Unseen floor depth, true heights, contact and hidden limbs remain uncertain.

## Clothing and objects

garment_parser.py runs a pinned MIT SCHP ATR-18 quantized ONNX model. Per-person crops produce broad garment/skin regions. Pose-local measurements derive sleeve/trouser coverage, hem and flare; background is unknown rather than bare skin. Semantic categories distinguish upper garments, pants, skirts, dresses, headwear, bags, scarves and glasses.

appearance.py combines these regions with local CLIP image/text similarity cues for fine supported categories such as suits, bridal dresses and bouquets. Conflicting or unsupported evidence produces notes rather than invented certainty. Raw scores are not calibrated probabilities. Strongly recognized unsupported wheelchairs/canes become blockers; other recognized unsupported objects produce warnings. The recognizer can still miss objects or misclassify clothing; it is not universal reconstruction.

Appearance.garment stores inferred upper/lower categories and bounded length/flare values. Unknown fields stay explicit on fresh and reopened projects. Figure.analysis_blockers persists major unresolved objects; analysis_warnings stores other uncertainty. Garment geometry follows the same joint frames/limb chains, with real shirt hems, sleeves, trouser legs and skirt/dress drapes. Added hats, bags, scarves and glasses are fused solids. No photo texture substitutes for actual geometry.

Models are installed explicitly, never downloaded by inference. YOLOX and SCHP use pinned revisions plus hardcoded SHA-256; CLIP verifies upstream checksums and a local manifest. The SCHP MIT notice is packaged. All inference is local CPU; no photo reaches a model provider, paid API or telemetry service.

## Geometry and coordinates

Body/STL coordinates are right-handed Z-up millimeters, with default front toward negative Y. A smooth signed-distance body sampled at 0.55 template-mm becomes a closed marching-cubes surface; Manifold combines real garments/accessories. Each figure is normalized to its target height, then receives uniform scale, Z rotation and XY placement. Pale green material and tiny dark recess material groups affect previews/GLB; STL retains actual facial recesses without color.

The viewport receives local figure meshes and reapplies the same transforms used for exports. Shared pedestal geometry remains world-space under reserved ID __group_base__. The read-only photo viewport exposes orbit/measurement, reports bounds of the complete scene and keeps all included figures visible. Whole-scene sizing scales every figure and XY spacing together; relative proportions remain.

A shared base suppresses individual bases, lifts figures slightly into its surface and includes fused raised inscription. Group export unions actual solids; without a shared base, several separate closed solids may remain in one STL. Final float32 STL reimport must retain watertightness, winding and positive volume. Invalid topology is rejected rather than hidden with automatic hole filling.

Mesh checks cover closed edge incidence, winding, positive volume, degenerates, components, approximate structural feature sizes and planar contact stability. They do not solve supports, complete local wall thickness, overhangs, slicing, drainage or print success. Measurements are Euclidean, not geodesic. GLB converts mm/Z-up to meters/Y-up.

## Sharing and operation

PNG captures only the 3D canvas. WebM records an actual eight-second orbit with browser capability checks. GLB exports visible group meshes/materials. Self-contained HTML embeds safely escaped names/geometry and a plain WebGL viewer with no external libraries or network access; it contains no reference photo and creates no hosted URL.

Uploads are bounded before multipart parsing, limited to JPEG/PNG/WEBP, 10 MB and 20 MP, EXIF-oriented and resized within 1600 px. Photo inference is serialized; geometry concurrency, pending jobs, retained results and retention time are bounded. Jobs are in-memory and disappear on restart. This is a local single-user app, not a public multi-tenant service.

The Windows ZIP includes compiled assets. The batch launcher finds Python 3.12, prepares a private venv, installs dependencies/models, starts a loopback-only FastAPI server and opens /welcome/ after app/version health checks. Existing version-2 servers are not reused as version 3. Native Windows execution remains unverified in the Linux test environment.
