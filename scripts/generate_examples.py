from pathlib import Path
import json
from backend.app.schemas import Figure, Project
from backend.app.geometry import generate, joints
root=Path(__file__).resolve().parents[1]
f=Figure(id='demo-standing',name='Little figure 1',height_mm=100)
p=Project(figures=[f])
(root/'examples/standing-100mm.stl').write_bytes(generate(f).export(file_type='stl'))
(root/'examples/standing.project.json').write_text(p.model_dump_json(indent=2)+'\n')
(root/'examples/manual-poses.json').write_text(json.dumps({
    'description':'Original manual skeletons. Not inferred from photos. Canonical template coordinates; Z up.',
    'poses':{pose:{k:v.tolist() for k,v in joints(Figure(id='sample',pose=pose)).items()}
    for pose in ['standing','seated','reaching','leaning','crouching','arms-raised']}
},indent=2)+'\n')
print('Generated 100 mm STL, versioned project, and manual sample pose data.')
