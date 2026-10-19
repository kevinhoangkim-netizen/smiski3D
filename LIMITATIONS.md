# Limits and verification scope

## What photo-driven means

The app estimates a complete stylized group scene from visible people and broad clothing regions. It is not an exact scan, a general-purpose image-to-3D foundation model, or a reconstruction of every garment/object. Fine fabric, lace, patterns, jewelry, hidden surfaces and accurate physical likeness are outside its capabilities. All characters share an original rounded pale green style; no official character sculpt is included.

The new flow removes pose/outfit/accessory menus. It does not give the underlying vision models perfect understanding. Clothing classifications, measured lengths and accessory ownership can still be wrong, even on clear photos. Notes must remain visible; matching person counts alone is insufficient validation.

## Group detection and pose

Independent YOLOX body detection plus per-person MediaPipe analysis improves groups that the earlier pose-only approach reduced to one or two people. Small, faded, cropped, crowded, silhouetted, seated or overlapping subjects remain difficult. Six is a supported cap, not a detection guarantee. Review outlines and confirm the actual count; a person missed by every detector cannot automatically be marked unresolved.

Detected people without a usable pose remain visible as unresolved regions and block generation. Over-limit photos also block partial scenes. Marking a missing person triggers additive recovery; a failure must remain explicit rather than accepting an existing overlapping person as new evidence. Recovery is not an instruction to invent a person inside any rectangle.

Single-view depth, original heights, exact floor placement and hidden limbs are ambiguous. Geometry preserves estimated visible directions using canonical stylized lengths; hand contact and hugging interactions can shift. Automatic horizontal spacing separates enlarged heads without changing limb directions; it cannot preserve every contact while also changing human proportions. Back-facing shoulder order controls the face surface, but side views and inconsistent landmarks can still yield uncertain orientation. Elevated people, benches, wheelchairs and other scene supports need capabilities beyond a common ground-plane figurine arrangement.

## Clothing and accessories

Local SCHP semantic garment regions and CLIP cues inform simplified clothing solids. Measurements must distinguish actual skin from unknown/background; absent mask evidence must not shorten sleeves or remove clothes. Overlapping people, coats, long tops, crossed limbs and low contrast can contaminate regions. Hems, cuffs, lapels, folds, flower petals and accessory details are modeled solids, but fine cuts, patterns, lace and unseen backs are approximated.

Supported broad forms: shirts/jackets, pants/shorts, skirts/dresses; cap/top hat/wide brim/beanie; bouquet, veil, bow tie, bag, scarf and glasses. Props use thick printable shapes rather than detailed cloth/flower simulation. Other objects can be absent if unrecognized. Strongly recognized unsupported wheelchairs/canes block generation, but object recognition itself is fallible. Unfamiliar supported-looking objects can also be misclassified.

Unknown upper/lower categories are estimated automatically using available local evidence, known outfit relationships and measured coverage. When evidence is unavailable, simple shirt/trouser shapes are used and explicitly labeled estimated. These choices can be wrong; the estimates are not recognition confidence. Known categories are retained. Unknown dimensions use bounded approximations. The app does not provide manual wardrobe overrides in the photo flow.

## Real-photo validation

Local fixtures include real distinct-person families, hikers, seated beach friends, formal groups, accessibility aids, silhouettes, damaged photos and over-limit groups, with source/license provenance and visual expectations in ignored .local/group-validation-v3. Raw photos are not packaged or committed. Older tiled copies only demonstrate plumbing and are not an accuracy benchmark.

Successful counts/poses and known failures must be reported separately. A six-person historical family was counted as six but initially resolved only four poses; that case is blocked. Faded/backlit/cropped groups can also remain unresolved. Visual skeleton review caught a collapsed body despite high model visibility, and clothing review caught misleading pants/skirt/sleeve estimates. These checks drive guards and disclose limits rather than supporting a universal accuracy claim.

Automated browser tests mock some analysis responses to verify UI contracts. Python tests cover integrity, decisions, geometry and API enforcement. Neither proves broad recognition accuracy. Representative real inference and exact exported STL reimport are checked separately.

## Printing, saving and sharing

Watertightness, winding and dimension checks do not guarantee print success. Feature thickness/stability are heuristics, not full wall-thickness/load/overhang analysis. There is no support generation, slicing, hollowing, drainage or printer-specific fit. Inspect orientation, build volume, material and supports in your slicer. STL has neither colors nor units; select millimeters. Glow requires suitable physical material.

Browser autosave is local to its origin, not cloud backup. Clearing storage or changing ports can hide a draft. Save important JSON files. Photos are excluded, but names, pose coordinates and measurements can still be personal information. Shared HTML/GLB contain names/geometry. HTML attachments may be restricted by email/corporate policy; PNG/video are alternatives. WebM requires browser recording support.

## Platform and hosting

The app, launcher, CPU models, build and browser checks run on Linux/Python 3.12/Chromium. A combined natural-photo inference pass peaked around 1.18 GiB of process memory; scene complexity and mesh caching can use more. Native Windows execution remains untested; compatible dependency wheels exist. The ZIP includes the built website and needs no Node.js. First setup downloads roughly 255 MiB of models plus dependencies; later launches reuse them. No paid API, account or GPU is required.

No public website URL, gallery, login, payments or ordering service is included. Sharing creates files. Public multi-tenant hosting would need authentication, isolation, quotas, scheduling, durable storage and operational work.


Visible joint mapping preserves image-plane proportions and foreshortening, but model landmark errors, hidden-joint fallback, stylized head enlargement and independent grounding can still differ from the original pose or group contact. Depth and elevated seating/props cannot be recovered reliably from one view.

Print bases provide flat fused supports and keep selected physical thickness while resizing. Stability checks use the mass projection/contact hull only; they are not a guarantee of printability. Thin detail, overhangs, material choice, supports and slicer settings still matter. Native Windows execution has not been available for testing; updater file integrity/preservation/recovery and local shutdown are exercised in Linux tests.
