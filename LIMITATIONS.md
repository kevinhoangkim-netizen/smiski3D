# Limits and verification scope

## Photo interpretation

A single photograph cannot establish true height, hidden limbs, exact floor depth or unseen garment surfaces. Pose fitting preserves visible image-space gesture using stylized anatomical lengths. Relative heights/layout are estimates; portrait/landscape normalization is corrected using the saved image aspect. Older photo projects without this field use the square-image legacy default and should be reanalyzed for improved directions.

Six people is a cap, not a detection guarantee. Overlapping-window inference can still miss small/occluded people or produce duplicates in crowded scenes. Subject inclusion and corrections must be reviewed. A six-copy collage of the public MediaPipe sample demonstrates multi-person plumbing; it does not represent a diverse natural group benchmark. No identity or likeness reconstruction is promised.

## Automatic clothing and accessories

Local CLIP compares image crops with a fixed supported category set. It is not an instance-segmentation system, a garment pattern reconstructor, or a universally accurate accessory detector. Close/overlapping people can contaminate crops. Small hats/flowers/ties and transparent veils can be uncertain or missed. Unsupported garments and weak comparisons abstain rather than being called confident detections. Raw similarity is not a calibrated probability.

Supported printable forms: classic/casual/suit/wedding dress; cap/top hat/wide brim; bouquet, veil, bow tie. Other clothing requires future templates. Bouquets and veils use chunky supported geometry, not individual flower petals or cloth physics. Automatic suggestions are editable, and changing an occasion does not overwrite the detected people with arbitrary wedding outfits.

The roughly 149 MiB of local classifier weights are downloaded separately with checksum verification. The measured CPU process peaked around 635 MiB of memory in the small validation set. Pose/model runtime failures are explicit. If installation is blocked, pose/manual editing can continue; this does not mean automatic clothing recognition is working.

Real-image spot checks recognized a bride's wedding dress and bouquet in a hugging couple, and a suit in a separate full-body wedding photo. The partly hidden groom abstained; bouquet ownership was ambiguous in the other photo. Direct hat/bow-tie crops abstained, and a cropped three-person wedding photo yielded only one person. These are useful failure examples, not an accuracy benchmark. A six-person check used tiled copies of the pose sample rather than six distinct people.

## Geometry and printing

The figures follow the supplied high-level rounded style but are original procedural sculpts. No proprietary mesh or official sculpt/packaging is used. Finite sampling/simplification can be visible at close zoom. Tiny facial marks are physical shallow recesses; dark paint-like material is only in the preview/GLB. STL has neither colors nor units. A physical glowing figure requires an appropriate material.

Mesh tests do not guarantee print success. Thickness and stability are heuristics, not complete wall-thickness/overhang/load analysis. No supports, slicing, toolpaths, hollowing or drainage are generated. Inspect supports, printer build volume, resin/material behavior and orientation in a slicer. Shared base and inscription can increase total scene dimensions; no silent scale-to-printer is applied. Separate figures may intentionally export as disjoint closed solids.

## Local saving and sharing

Autosave is browser-local, not a cloud backup. Clearing browser storage or changing the server origin/port can remove or hide that draft. Explicit JSON saving is recommended for important projects. Photos are excluded; landmark coordinates and figure names can still be personal information. Remove the reference photo and/or delete the draft using their separate controls.

Sharing exports files, not hosted URLs. The self-contained HTML viewer includes figure names and geometry; recipients must open it in a WebGL-capable browser. Corporate/email policies may restrict HTML attachments. PNG/video are alternatives. WebM requires browser MediaRecorder/canvas recording support; actual playback/export behavior varies by browser. Axis buttons are aligned perspective views, not orthographic cameras.

## Platform and operation

The app, geometry, browser and launcher checks run on Linux CPU, Python 3.12 and Chromium. Native Windows execution was not tested, although compatible Python 3.12 Windows x64 wheels were verified for the core native dependencies and ONNX/tokenizers. The prepared ZIP includes built assets and does not require Node.js. First setup needs Internet access for dependencies/models; no paid API/account is required.

This is a local single-user app. Public multi-tenant deployment would still require authentication, isolation, quotas, upload scheduling, durable storage and hosting. No login, public gallery, payments, anonymous hosted project URL or ordering service is included.

## Most useful next improvements

1. Build a consented regression set of distinct-person weddings/families, difficult occlusion and natural six-person photos; measure misses, duplicates and outfit/accessory accuracy.
2. Add better per-person segmentation and fine correction controls for hands/head direction and contact gestures.
3. Add full local thickness/overhang analysis and printer-specific build-volume guidance.
4. Validate the installer and end-to-end flow on native Windows hardware.
