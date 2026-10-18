"""Original parametric figures. All model and export coordinates are Z-up millimeters."""
from functools import lru_cache
import math
import numpy as np
import trimesh
from .schemas import Figure, Project

PRESETS = ['standing', 'seated', 'reaching', 'leaning', 'crouching', 'arms-raised']

def joints(figure):
    j = dict(hip_l=[-7,0,31], hip_r=[7,0,31], shoulder_l=[-11,0,56], shoulder_r=[11,0,56],
             elbow_l=[-17,0,43], elbow_r=[17,0,43], wrist_l=[-15,-2,32], wrist_r=[15,-2,32],
             knee_l=[-7,0,17], knee_r=[7,0,17], ankle_l=[-7,0,5], ankle_r=[7,0,5])
    if figure.pose == 'arms-raised':
        j.update(elbow_l=[-21,0,68], elbow_r=[21,0,68], wrist_l=[-24,0,81], wrist_r=[24,0,81])
    elif figure.pose == 'reaching':
        j.update(elbow_r=[22,-3,61], wrist_r=[31,-5,72], elbow_l=[-17,-2,44], wrist_l=[-23,-6,36])
    elif figure.pose == 'leaning':
        for key in j:
            j[key][0] += max(0,j[key][2]-10)*.18
        j.update(elbow_l=[-13,-2,48], wrist_l=[-23,-4,54])
    elif figure.pose in ('seated','crouching'):
        seated = figure.pose == 'seated'
        for key in ('hip_l','hip_r','shoulder_l','shoulder_r','elbow_l','elbow_r','wrist_l','wrist_r'):
            j[key][2] -= 13 if seated else 17
        for side, sign in [('l',-1),('r',1)]:
            j['knee_'+side]=[sign*9,-17,18 if seated else 15]
            j['ankle_'+side]=[sign*10,-18 if seated else -2,5]
            j['wrist_'+side]=[sign*10,-17,25]
    elif figure.pose == 'photo':
        lm = figure.landmarks
        left_sign=1 if min(lm[11].visibility,lm[12].visibility)>=.5 and lm[11].x>=lm[12].x else -1
        for side,sign in [('l',left_sign),('r',-left_sign)]:
            j['shoulder_'+side][0]=sign*11
            j['hip_'+side][0]=sign*7
        # Root centered, canonical torso and limb lengths. Depth is deliberately
        # conservative; image-space direction is preserved, identity is not inferred.
        mapping = {'shoulder_l':11,'shoulder_r':12,'elbow_l':13,'elbow_r':14,
                   'wrist_l':15,'wrist_r':16,'hip_l':23,'hip_r':24,
                   'knee_l':25,'knee_r':26,'ankle_l':27,'ankle_r':28}
        for side in ('l','r'):
            for parent, child, length in [('shoulder','elbow',16),('elbow','wrist',14),('hip','knee',17),('knee','ankle',15)]:
                p, c = parent+'_'+side, child+'_'+side
                start, end = lm[mapping[p]], lm[mapping[c]]
                if min(start.visibility, end.visibility) >= .5:
                    v = np.array([end.x-start.x, np.clip((end.z-start.z)*.35,-.15,.15), -(end.y-start.y)])
                    if np.linalg.norm(v) > .015:
                        j[c] = (np.array(j[p]) + v/np.linalg.norm(v)*length).tolist()
    return {k:np.array(v,float) for k,v in j.items()}

def body_surface(figure, j):
    """Smooth implicit union, meshed as a single closed original surface."""
    from skimage.measure import marching_cubes
    hip=(j['hip_l']+j['hip_r'])/2
    shoulder=(j['shoulder_l']+j['shoulder_r'])/2
    head=shoulder+[0,-.3,23]
    shapes=[]
    def ell(c,r): shapes.append(('ell',np.asarray(c,float),np.asarray(r,float)))
    def cap(a,b,r): shapes.append(('cap',np.asarray(a,float),np.asarray(b,float),r))
    ell((hip+shoulder)/2,[13,9,19]);ell(hip,[12,9,10])
    cap(shoulder,shoulder+[0,0,7],6)
    ell(head,[14*figure.head_size,12*figure.head_size,15*figure.head_size])
    for side in ('l','r'):
        for a,b,r in [('shoulder','elbow',4.6),('elbow','wrist',4.2),('hip','knee',5.7),('knee','ankle',5.2)]:
            cap(j[a+'_'+side],j[b+'_'+side],r)
        ell(j['wrist_'+side],[4.8,4.8,5.2])
        ell(j['ankle_'+side]+[0,-3,-1.5],[6.5,10,4])
    bounds=[]
    for shape in shapes:
        if shape[0]=='ell': bounds += [shape[1]-shape[2],shape[1]+shape[2]]
        else: bounds += [shape[1]-shape[3],shape[1]+shape[3],shape[2]-shape[3],shape[2]+shape[3]]
    low=np.min(bounds,axis=0)-3
    high=np.max(bounds,axis=0)+3
    spacing=.65
    axes=[np.arange(low[i],high[i]+spacing,spacing,dtype=np.float32) for i in range(3)]
    x,y,z=np.ix_(*axes)
    def ell_field(c,r):
        # Standard first-order ellipsoid distance estimate, smooth at the surface.
        v=[x-c[0],y-c[1],z-c[2]]
        k0=np.sqrt(sum((v[i]/r[i])**2 for i in range(3)))
        k1=np.sqrt(sum((v[i]/(r[i]*r[i]))**2 for i in range(3)))
        return np.where(k0<1e-6,-min(r),k0*(k0-1)/np.maximum(k1,1e-8))
    def smooth_union(a,b,k=2.3):
        h=np.maximum(k-abs(a-b),0)/k
        return np.minimum(a,b)-h*h*k*.25
    field=None
    for shape in shapes:
        if shape[0]=='ell': d=ell_field(shape[1],shape[2])
        else:
            a,b,r=shape[1:];v=b-a
            t=np.clip(((x-a[0])*v[0]+(y-a[1])*v[1]+(z-a[2])*v[2])/max(float(v@v),1e-8),0,1)
            d=np.sqrt((x-a[0]-t*v[0])**2+(y-a[1]-t*v[1])**2+(z-a[2]-t*v[2])**2)-r
        field=d if field is None else smooth_union(field,d)
    nose=ell_field(head+[0,-11.7*figure.head_size,-1],[1.6,2,2])
    field=smooth_union(field,nose,.6)
    for eye_x in [-4,4]:
        eye=ell_field(head+[eye_x,-11.5*figure.head_size,2.5],[1.1,1.8,1.1])
        field=np.maximum(field,-eye)
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=(spacing,)*3,gradient_direction='ascent')
    mesh=trimesh.Trimesh(vertices=vertices+low,faces=faces)
    if mesh.volume<0: mesh.invert()
    # Reduce redundant near-planar triangles without visible silhouette changes.
    from manifold3d import Manifold, Mesh
    solid=Manifold(Mesh(vert_properties=np.asarray(mesh.vertices,dtype=np.float32),tri_verts=np.asarray(mesh.faces,dtype=np.uint32)))
    raw=solid.simplify(.025).to_mesh64()
    return stl_precision_mesh(trimesh.Trimesh(vertices=raw.vert_properties,faces=raw.tri_verts))

@lru_cache(maxsize=48)
def _generate(serialized):
    figure = Figure.model_validate_json(serialized)
    j = joints(figure)
    mesh=body_surface(figure,j)
    mesh.apply_translation([0,0,-mesh.bounds[0,2]])
    if figure.base.shape != 'none':
        height = figure.base.thickness_mm
        # Slight overlap ensures a true union with the feet.
        mesh.apply_translation([0,0,height-.5])
        bounds=mesh.bounds.copy()
        center=(bounds[0,:2]+bounds[1,:2])/2
        size=bounds[1,:2]-bounds[0,:2]+2*figure.base.margin_mm
        if figure.base.shape=='round':
            radius=np.linalg.norm(size/2)
            base=trimesh.creation.cylinder(radius=radius,height=height,sections=64)
        else:
            base=trimesh.creation.box(extents=[*size,height])
        base.apply_translation([*center,height/2])
        mesh=trimesh.boolean.union([mesh,base],engine='manifold')
    factor=figure.height_mm/mesh.extents[2]
    mesh.apply_scale(factor)
    mesh.metadata['structural_diameter_mm']=8.4*factor
    mesh.remove_unreferenced_vertices()
    return mesh

def generate(figure: Figure, transformed=True):
    # Transform changes reuse expensive boolean geometry.
    raw=figure.model_copy(update={'transform': type(figure.transform)()})
    mesh=_generate(raw.model_dump_json()).copy()
    if transformed:
        t=figure.transform
        mesh.apply_scale(t.scale)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(t.rotation),[0,0,1]))
        mesh.apply_translation([t.x,t.y,0])
    return mesh

def project_meshes(project: Project):
    return [(f,generate(f)) for f in project.figures if f.included]

def dimensions(mesh):
    return {k:round(float(v),3) for k,v in zip(['width','depth','height'],mesh.extents)}

def measure(a,b):
    return float(np.linalg.norm(np.asarray(a,float)-np.asarray(b,float)))

def validate(mesh, minimum_feature_mm=1.2, feature_estimate_mm=None):
    areas=mesh.area_faces
    edges=mesh.edges_sorted
    _, counts=np.unique(edges,axis=0,return_counts=True)
    components=len(mesh.split(only_watertight=False))
    manifold=bool(len(counts)>0 and np.all(counts==2))
    degenerate=int(np.sum(areas<1e-9))
    items=[]
    def add(name,status,detail): items.append(dict(name=name,status=status,detail=detail))
    add('Watertight','pass' if mesh.is_watertight else 'fail','Every surface edge must be shared by exactly two faces.')
    add('Manifold edges','pass' if manifold else 'fail','Edge incidence check; does not prove absence of every vertex defect.')
    add('Consistent winding','pass' if mesh.is_winding_consistent and mesh.volume>0 else 'fail','Outward, consistent triangle winding and positive signed volume.')
    add('Degenerate faces','pass' if degenerate==0 else 'fail',f'{degenerate} near-zero-area triangles.')
    add('Connected components','pass' if components==1 else 'warning',f'{components} solid component(s). Separate group figures may intentionally remain separate.')
    # Parameter-derived structural estimate, explicitly not a local thickness solver.
    if feature_estimate_mm is None:
        add('Feature thickness','warning','No reliable local wall-thickness estimate for this arbitrary mesh.')
    else:
        add('Feature thickness','warning' if feature_estimate_mm<minimum_feature_mm else 'pass',
            f'Approx. smallest structural limb diameter {feature_estimate_mm:.1f} mm; threshold {minimum_feature_mm:.1f} mm. Eyes, joints, overhangs and every local wall are not measured.')
    bottom=mesh.bounds[0,2]
    faces=mesh.triangles
    contact=np.all(abs(faces[:,:,2]-bottom)<.02,axis=1)
    area=float(np.sum(areas[contact]))
    contact_vertices=faces[contact].reshape(-1,3) if np.any(contact) else np.empty((0,3))
    stable=False
    if len(contact_vertices)>=3:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull=ConvexHull(contact_vertices[:,:2])
            com=mesh.center_mass[:2]
            stable=bool(np.all(hull.equations[:,:2]@com+hull.equations[:,2] <= -.25))
        except QhullError:
            pass
    add('Build-plate stability','pass' if stable and area>20 else 'warning',
        f'Flat contact area {area:.1f} mm². '+('Mass projection lies within contact hull. ' if stable else 'Small contact or mass near/outside support. ')+ 'Heuristic only; inspect in your slicer. Supports may still be needed.')
    return dict(status='fail' if any(i['status']=='fail' for i in items) else 'warning' if any(i['status']=='warning' for i in items) else 'pass',
                checks=items,dimensions=dimensions(mesh),triangles=len(mesh.faces),components=components,
                caveat='Mesh checks cannot guarantee printing success. Supports, orientation, resin drainage and slicing remain your responsibility.')

def figure_report(figure,mesh,threshold):
    feature=generate(figure,False).metadata['structural_diameter_mm']*figure.transform.scale
    report=validate(mesh,threshold,feature)
    if figure.pose=='photo':
        uncertain=sum(p.visibility<.5 for p in figure.landmarks)
        report['checks'].append(dict(name='Pose interpretation',status='warning',detail=f'Estimated single-photo directions with canonical anatomy. {uncertain} landmarks below 50% visibility; neutral limb fallbacks may be used. Depth and hidden anatomy are uncertain.'))
        if report['status']=='pass': report['status']='warning'
    return report

def union_or_compose(meshes):
    # Preserve actual union semantics; simplify float32 intersection slivers before
    # STL vertex welding, which otherwise collapses tiny coplanar triangles.
    if len(meshes)==1:
        return stl_precision_mesh(meshes[0])
    from manifold3d import Manifold, Mesh
    solids=[Manifold(Mesh(vert_properties=np.asarray(m.vertices,dtype=np.float32),
                         tri_verts=np.asarray(m.faces,dtype=np.uint32))) for m in meshes]
    result=solids[0]
    for solid in solids[1:]:
        result=result+solid
    raw=result.set_tolerance(1e-4).simplify(1e-4).to_mesh64()
    return stl_precision_mesh(trimesh.Trimesh(vertices=raw.vert_properties,faces=raw.tri_verts))


def stl_precision_mesh(mesh):
    # Binary STL quantizes to float32. Weld at that precision before checking
    # topology, then drop only triangles collapsed by that quantization. Never
    # fill holes or hide remaining invalid topology.
    result=trimesh.Trimesh(vertices=np.asarray(mesh.vertices,dtype=np.float32),faces=mesh.faces)
    result.update_faces(result.area_faces>0)
    result.remove_unreferenced_vertices()
    return result
