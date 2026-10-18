"""Soft original figurine solids. All model/export coordinates are Z-up millimeters."""
from functools import lru_cache
import math
import numpy as np
import trimesh
from .schemas import Figure, Project

PRESETS = ['standing', 'seated', 'reaching', 'leaning', 'crouching', 'arms-raised']


def _unit(value, fallback=(0, 0, 1)):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    return value / length if length > 1e-8 else np.asarray(fallback, dtype=float)


def _plane_axis(candidate, normal):
    """An orthogonal axis, including poses with a completely horizontal hand."""
    normal=_unit(normal)
    projected=np.asarray(candidate,float)-normal*np.dot(candidate,normal)
    if np.linalg.norm(projected)<1e-6:
        candidate=np.eye(3)[np.argmin(np.abs(normal))]
        projected=candidate-normal*np.dot(candidate,normal)
    return _unit(projected)


def joints(figure):
    """A short, rounded anatomy with image-space gesture and torso direction.

    Photo landmarks supply directions, including the torso and bent knees, rather
    than merely moving hands on an upright stock torso. Hidden joints retain a
    documented neutral fallback; no hidden photo detail is asserted as certain.
    """
    j = dict(hip_l=[-7,0,31], hip_r=[7,0,31], shoulder_l=[-10,0,56], shoulder_r=[10,0,56],
             elbow_l=[-15,0,43], elbow_r=[15,0,43], wrist_l=[-14,-1,31], wrist_r=[14,-1,31],
             knee_l=[-7,0,18], knee_r=[7,0,18], ankle_l=[-7,0,5], ankle_r=[7,0,5])
    if figure.pose == 'arms-raised':
        j.update(elbow_l=[-21,0,68], elbow_r=[21,0,68], wrist_l=[-23,0,80], wrist_r=[23,0,80])
    elif figure.pose == 'reaching':
        j.update(elbow_r=[22,-3,61], wrist_r=[31,-5,71], elbow_l=[-16,-2,44], wrist_l=[-22,-6,35])
    elif figure.pose == 'leaning':
        for key in j:
            j[key][0] += max(0,j[key][2]-10)*.2
        j.update(elbow_l=[-13,-2,48], wrist_l=[-21,-4,54])
    elif figure.pose in ('seated','crouching'):
        seated = figure.pose == 'seated'
        for key in ('hip_l','hip_r','shoulder_l','shoulder_r','elbow_l','elbow_r','wrist_l','wrist_r'):
            j[key][2] -= 13 if seated else 17
        for side, sign in [('l',-1),('r',1)]:
            j['knee_'+side]=[sign*9,-17,18 if seated else 15]
            j['ankle_'+side]=[sign*10,-18 if seated else -2,5]
            j['wrist_'+side]=[sign*10,-17,25]
    j = {k: np.asarray(v, dtype=float) for k,v in j.items()}
    if figure.pose == 'photo':
        lm = figure.landmarks
        def point(index):
            p = lm[index]
            # MediaPipe normalizes X (and estimated Z) by image width, but Y
            # by height. Convert into equal image-height units before taking
            # directions; otherwise a landscape group flattens diagonal poses.
            aspect=getattr(figure,'photo_aspect',1)
            return np.array([p.x*aspect,np.clip(p.z*aspect*.3,-.2,.2),-p.y],dtype=float)
        def reliable(*indices):
            return min(lm[i].visibility for i in indices) >= .5
        shoulder_mid = (point(11)+point(12))/2
        hip_mid = (point(23)+point(24))/2
        direction_threshold=max(np.linalg.norm(shoulder_mid-hip_mid)*.025,1e-5)
        torso_up = _unit(shoulder_mid-hip_mid)
        if not reliable(11,12,23,24):
            torso_up = np.array([0.,0.,1.])
        # Keep the observed shoulder/hip slopes. Forcing those axes to be
        # perpendicular to the torso silently levels a twisted/leaning pose.
        lateral = _unit(point(12)-point(11), (1,0,0))
        hip_lateral = _unit(point(24)-point(23), lateral)
        root=np.array([0.,0.,31.])
        for side, sign in [('l',-1),('r',1)]:
            j['hip_'+side]=root+sign*hip_lateral*7
            j['shoulder_'+side]=root+torso_up*25+sign*lateral*10
        mapping={'shoulder_l':11,'shoulder_r':12,'elbow_l':13,'elbow_r':14,
                 'wrist_l':15,'wrist_r':16,'hip_l':23,'hip_r':24,
                 'knee_l':25,'knee_r':26,'ankle_l':27,'ankle_r':28}
        for side in ('l','r'):
            for parent,child,length in [('shoulder','elbow',14),('elbow','wrist',12),('hip','knee',15),('knee','ankle',13)]:
                p,c=parent+'_'+side,child+'_'+side
                if reliable(mapping[p],mapping[c]):
                    direction=point(mapping[c])-point(mapping[p])
                    if np.linalg.norm(direction)>direction_threshold:
                        j[c]=j[p]+_unit(direction)*length
                    else:
                        j[c]=j[p]-torso_up*length
                else:
                    j[c]=j[p]-torso_up*length
        face_indices=[i for i in (0,7,8) if lm[i].visibility>=.5]
        head_up=torso_up
        if face_indices and reliable(11,12):
            face_mid=np.mean([point(i) for i in face_indices],axis=0)
            head_up=_unit(face_mid-shoulder_mid,torso_up)
            # A small face-depth estimate must not turn the whole head sideways.
            head_up=_unit([head_up[0], np.clip(head_up[1],-.25,.25),head_up[2]],torso_up)
        j['head_up']=head_up
        if reliable(2,5):
            j['head_lateral']=_unit(point(5)-point(2),(1,0,0))
    return j


def body_frame(j):
    hip=(j['hip_l']+j['hip_r'])/2
    shoulder=(j['shoulder_l']+j['shoulder_r'])/2
    up=_unit(shoulder-hip)
    lateral=_unit(j['shoulder_r']-j['shoulder_l'],(1,0,0))
    if lateral[0]<0:
        lateral=-lateral
    lateral=_plane_axis(lateral,up)
    forward=_unit(np.cross(lateral,up),(0,-1,0))
    return hip,shoulder,up,lateral,forward


def body_surface(figure, j):
    """One smooth implicit surface: large head, belly, small feet, no nose."""
    from skimage.measure import marching_cubes
    hip,shoulder,up,lateral,forward=body_frame(j)
    head_up=_unit(j.get('head_up',up),up)
    head=shoulder+head_up*18
    head_lateral=_unit(j.get('head_lateral',lateral),lateral)
    if np.dot(head_lateral,lateral)<0:
        head_lateral=-head_lateral
    head_lateral=_plane_axis(head_lateral,head_up)
    head_forward=_unit(np.cross(head_lateral,head_up),forward)
    frame=np.stack([lateral,forward,up],axis=1)
    head_frame=np.stack([head_lateral,head_forward,head_up],axis=1)
    shapes=[]
    def ell(c,r,axes=frame):
        shapes.append(('ell',np.asarray(c,float),np.asarray(r,float),axes))
    ell((hip+shoulder)/2,[11.7,8.3,16])
    ell(hip+up*7+forward*1.2,[12,9.8,12])
    ell(shoulder-up*2,[10.7,7,5])
    ell(head,[18.5*figure.head_size,17.2*figure.head_size,19*figure.head_size],head_frame)
    for side in ('l','r'):
        # Each limb is one rounded chain. Blending each segment separately
        # would add a visible ball at the elbow/knee even without joint spheres.
        for chain,radius in [(('shoulder','elbow','wrist'),3.7),(('hip','knee','ankle'),5.4)]:
            shapes.append(('chain',np.asarray([j[key+'_'+side] for key in chain]),radius))
        # A smooth taper, with no spherical knuckle/hand joint or shoe shape.
        hand_direction=_unit(j['wrist_'+side]-j['elbow_'+side],-up)
        hand_lateral=_plane_axis(lateral,hand_direction)
        hand_axes=np.stack([hand_lateral,_unit(np.cross(hand_lateral,hand_direction),forward),hand_direction],axis=1)
        ell(j['wrist_'+side]+hand_direction*.65,[3.7,3.5,4.1],hand_axes)
        ell(j['ankle_'+side]+[0,-1.4,-1.5],[5.1,6.4,3.5],np.eye(3))
    bounds=[]
    for shape in shapes:
        if shape[0]=='ell':
            extent=np.abs(shape[3])@shape[2]
            bounds += [shape[1]-extent,shape[1]+extent]
        else:
            bounds.extend(shape[1]-shape[2])
            bounds.extend(shape[1]+shape[2])
    low=np.min(bounds,axis=0)-3
    high=np.max(bounds,axis=0)+3
    # Sub-millimeter sampling retains round facial cavities in the real STL.
    spacing=.55
    axes=[np.arange(low[i],high[i]+spacing,spacing,dtype=np.float32) for i in range(3)]
    x,y,z=np.ix_(*axes)
    def ell_field(c,r,axes):
        world=[x-c[0],y-c[1],z-c[2]]
        v=[sum(world[k]*axes[k,i] for k in range(3)) for i in range(3)]
        k0=np.sqrt(sum((v[i]/r[i])**2 for i in range(3)))
        k1=np.sqrt(sum((v[i]/(r[i]*r[i]))**2 for i in range(3)))
        return np.where(k0<1e-6,-min(r),k0*(k0-1)/np.maximum(k1,1e-8))
    def smooth_union(a,b,k=1.8):
        h=np.maximum(k-abs(a-b),0)/k
        return np.minimum(a,b)-h*h*k*.25
    def capsule_field(a,b,r):
        v=b-a
        t=np.clip(((x-a[0])*v[0]+(y-a[1])*v[1]+(z-a[2])*v[2])/max(float(v@v),1e-8),0,1)
        return np.sqrt((x-a[0]-t*v[0])**2+(y-a[1]-t*v[1])**2+(z-a[2]-t*v[2])**2)-r
    field=None
    for shape in shapes:
        if shape[0]=='ell':
            d=ell_field(shape[1],shape[2],shape[3])
        else:
            points,radius=shape[1:]
            d=capsule_field(points[0],points[1],radius)
            for a,b in zip(points[1:-1],points[2:]):
                d=np.minimum(d,capsule_field(a,b,radius))
        field=d if field is None else smooth_union(field,d)
    # Shallow inset facial marks. There is deliberately no projecting nose.
    face_features=[]
    for eye_x in [-5.1,5.1]:
        surface_depth=17.2*figure.head_size*np.sqrt(1-(eye_x/(18.5*figure.head_size))**2-(1.5/(19*figure.head_size))**2)
        center=head+head_lateral*eye_x+head_forward*(surface_depth+.45)+head_up*1.5
        radii=np.array([1.05,1.1,1.05])
        face_features.append((center,radii,head_frame))
        eye=ell_field(center,radii,head_frame)
        field=np.maximum(field,-eye)
    mouth_depth=17.2*figure.head_size*np.sqrt(1-(4.5/(19*figure.head_size))**2)
    mouth_center=head+head_forward*(mouth_depth+.35)-head_up*4.5
    mouth_radii=np.array([2.1,.85,.55])
    face_features.append((mouth_center,mouth_radii,head_frame))
    mouth=ell_field(mouth_center,mouth_radii,head_frame)
    field=np.maximum(field,-mouth)
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=(spacing,)*3,gradient_direction='ascent')
    mesh=trimesh.Trimesh(vertices=vertices+low,faces=faces)
    if mesh.volume<0:
        mesh.invert()
    from manifold3d import Manifold, Mesh
    solid=Manifold(Mesh(vert_properties=np.asarray(mesh.vertices,dtype=np.float32),tri_verts=np.asarray(mesh.faces,dtype=np.uint32)))
    raw=solid.simplify(.025).to_mesh64()
    mesh=stl_precision_mesh(trimesh.Trimesh(vertices=raw.vert_properties,faces=raw.tri_verts))
    from .figure_style import accessory_meshes
    decorations=accessory_meshes(figure,j,head,up,lateral,forward)
    mesh=union_or_compose([mesh,*decorations]) if decorations else mesh
    mesh.metadata['face_features']=face_features
    return mesh


@lru_cache(maxsize=48)
def _generate(serialized):
    figure=Figure.model_validate_json(serialized)
    mesh=body_surface(figure,joints(figure))
    face_features=mesh.metadata.pop('face_features',[])
    vertical_shift=-mesh.bounds[0,2]
    mesh.apply_translation([0,0,vertical_shift])
    if figure.base.shape!='none':
        height=figure.base.thickness_mm
        mesh.apply_translation([0,0,height-.5])
        vertical_shift+=height-.5
        bounds=mesh.bounds.copy()
        center=(bounds[0,:2]+bounds[1,:2])/2
        size=bounds[1,:2]-bounds[0,:2]+2*figure.base.margin_mm
        if figure.base.shape=='round':
            base=trimesh.creation.cylinder(radius=np.linalg.norm(size/2),height=height,sections=64)
        else:
            base=trimesh.creation.box(extents=[*size,height])
        base.apply_translation([*center,height/2])
        mesh=union_or_compose([mesh,base])
    factor=figure.height_mm/mesh.extents[2]
    mesh.apply_scale(factor)
    feature_size=7.0
    accessories=set(figure.appearance.accessories)
    if 'bouquet' in accessories:
        feature_size=min(feature_size,5.0)
    if 'hat' in accessories:
        feature_size=min(feature_size,5.0 if figure.appearance.hat_style=='cap' else 3.4)
    mesh.metadata['structural_diameter_mm']=feature_size*factor
    mesh.remove_unreferenced_vertices()
    # Material labels highlight only the physically recessed facial surfaces.
    # STL stays monochrome and includes the same real cavities; no decals or
    # disconnected geometry are added for the preview's tiny dark marks.
    groups=np.zeros(len(mesh.faces),dtype=np.uint8)
    centers=mesh.triangles_center
    for center,radii,axes in face_features:
        center=(center+np.array([0,0,vertical_shift]))*factor
        local=(centers-center)@axes
        normalized=local/(radii*factor)
        inset=(np.sum(normalized**2,axis=1)<1.05)&(normalized[:,1]<.25)
        groups[inset]=1
    mesh.metadata['face_groups']=groups.tolist()
    return mesh


def generate(figure: Figure, transformed=True):
    # Geometry is reusable across people, names, analysis labels and sizes.
    # Uniform sizing happens after the cache, preserving exact target height.
    raw=figure.model_copy(update={'id':'geometry','name':'Figure','included':True,
        'height_mm':90,'transform':type(figure.transform)(),
        'appearance_source':'manual','appearance_confidence':None,'analysis_warnings':[]})
    mesh=_generate(raw.model_dump_json()).copy()
    size_factor=figure.height_mm/90
    mesh.apply_scale(size_factor)
    mesh.metadata['structural_diameter_mm']*=size_factor
    if transformed:
        t=figure.transform
        mesh.apply_scale(t.scale)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(t.rotation),[0,0,1]))
        mesh.apply_translation([t.x,t.y,0])
    return mesh


def project_meshes(project: Project):
    shared=getattr(project,'shared_base',None)
    pairs=[]
    for figure in project.figures:
        if not figure.included:
            continue
        actual=figure
        if shared and shared.shape!='none':
            actual=figure.model_copy(update={'base':figure.base.model_copy(update={'shape':'none'})})
        mesh=generate(actual)
        if shared and shared.shape!='none':
            mesh.apply_translation([0,0,shared.thickness_mm-.5])
        pairs.append((figure,mesh))
    return pairs


def project_scene(project: Project):
    """World-coordinate editable figures and optional real shared pedestal."""
    pairs=project_meshes(project)
    result=[(f.id,f.name,m,figure_report(f,m,project.minimum_feature_mm)) for f,m in pairs]
    shared=getattr(project,'shared_base',None)
    if shared and shared.shape!='none':
        from .pedestal import pedestal_mesh
        base=pedestal_mesh([m for _,m in pairs],shared)
        result.append(('__group_base__','Shared pedestal',base,validate(base,project.minimum_feature_mm)))
    return result


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
            f'Approx. smallest modeled structural feature {feature_estimate_mm:.1f} mm; threshold {minimum_feature_mm:.1f} mm. Facial recesses, joints, overhangs and every local wall are not measured.')
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
