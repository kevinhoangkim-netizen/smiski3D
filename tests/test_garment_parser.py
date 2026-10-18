"""Visible garment coverage, uncertainty and local parser integrity."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.app import appearance, garment_parser


def subject():
    points = [dict(x=.5, y=.1, z=0, visibility=1) for _ in range(33)]
    for index, xy in {11:(.3,.25),12:(.7,.25),13:(.25,.4),14:(.75,.4),
                      15:(.2,.55),16:(.8,.55),23:(.4,.5),24:(.6,.5),
                      25:(.4,.7),26:(.6,.7),27:(.4,.9),28:(.6,.9)}.items():
        points[index].update(x=xy[0], y=xy[1])
    return dict(landmarks=points, bbox=[.15,.08,.85,.94])


def clothing_mask(sleeve_end, trouser_end):
    image = Image.new('L', (200,400))
    draw = ImageDraw.Draw(image)
    draw.polygon([(60,100),(140,100),(120,205),(80,205)], fill=4)
    for side in (-1,1):
        shoulder=(100+side*40,100)
        wrist=(100+side*60,220)
        end=tuple(int(shoulder[k]+(wrist[k]-shoulder[k])*sleeve_end) for k in range(2))
        draw.line([shoulder,wrist],fill=2,width=12)
        draw.line([shoulder,end],fill=4,width=13)
        hip=(100+side*20,200)
        ankle=(100+side*20,360)
        hem=(hip[0],int(200+160*trouser_end))
        draw.line([hip,ankle],fill=2,width=17)
        draw.line([hip,hem],fill=4,width=18)
    return np.asarray(image)


def test_actual_mask_distinguishes_short_and_long_coverage():
    short=garment_parser.measurements(clothing_mask(.34,.45), subject(), (200,400),(0,0,200,400))
    long=garment_parser.measurements(clothing_mask(.95,.98), subject(), (200,400),(0,0,200,400))
    assert short['torso_clothes'] > .8
    assert .2 < short['sleeve_length'] < .55
    assert .3 < short['trouser_length'] < .6
    assert long['sleeve_length'] > .85
    assert long['trouser_length'] > .9


def test_neighbor_clothes_do_not_extend_own_bare_limbs():
    mask=clothing_mask(.25,.4).copy()
    mask[:,170:]=4
    result=garment_parser.measurements(mask,subject(),(200,400),(0,0,200,400))
    assert result['sleeve_length'] < .5
    assert result['trouser_length'] < .55


def test_occluded_limbs_are_unknown_not_full_length():
    person=subject()
    for index in (13,14,25,26): person['landmarks'][index]['visibility']=.1
    result=garment_parser.measurements(clothing_mask(.9,.9),person,(200,400),(0,0,200,400))
    assert result['sleeve_length'] is None
    assert result['trouser_length'] is None
    assert len(result['warnings'])==2


def test_background_beyond_sleeve_is_not_evidence_of_short_sleeves():
    mask=clothing_mask(.3,.4).copy()
    mask[mask==2]=0
    measured=garment_parser.measurements(mask,subject(),(200,400),(0,0,200,400))
    assert measured['sleeve_length'] is None
    assert measured['trouser_length'] is None


def test_preprocessing_preserves_shape_and_matches_original_schp_bgr():
    prepared=garment_parser.prepare_pixels(Image.new('RGB',(256,512),(255,0,0)))
    assert prepared.shape==(1,3,512,512)
    assert prepared[0,0,250,250]==pytest.approx((0-.406)/.225)
    assert prepared[0,2,250,250]==pytest.approx((1-.485)/.229)
    assert prepared[0,2,250,0]==pytest.approx((0-.485)/.229)


def test_unseen_clothes_near_limb_root_do_not_mean_zero_length():
    mask=clothing_mask(.95,.98).copy()
    mask[175:235,70:131]=0
    measured=garment_parser.measurements(mask,subject(),(200,400),(0,0,200,400))
    assert measured['trouser_length'] is None


def test_semantic_regions_recognize_trousers_and_not_neighbor_dress():
    labels=np.zeros((400,200),dtype=np.uint8)
    labels[95:205,60:141]=4
    labels[200:361,70:91]=6
    labels[200:361,110:131]=6
    labels[:,180:]=7
    evidence=garment_parser.semantic_evidence(labels,subject(),(200,400),(0,0,200,400))
    assert evidence['upper']=='shirt'
    assert evidence['lower']=='pants'


def test_independent_trouser_regions_override_false_dress_guess():
    value=appearance.default_appearance()
    value['outfit']='wedding-dress'
    warnings=[]
    appearance._garment_from_evidence(value,{},
        dict(torso_clothes=.9,torso_skin=0,sleeve_length=1,trouser_length=1,
             hem_length=None,skirt_flare=None,warnings=[],semantics={'upper':'shirt','lower':'pants'}),warnings)
    assert value['garment']['upper']=='shirt'
    assert value['garment']['lower']=='pants'
    assert value['outfit']=='casual'
    assert any('contradicted' in warning for warning in warnings)


def test_positive_mask_clothes_override_false_bare_label():
    value=appearance.default_appearance()
    warnings=[]
    appearance._garment_from_evidence(value, {'upper':('bare',True),'lower':('shorts',True)},
                                     dict(torso_clothes=.9,torso_skin=.02,sleeve_length=.3,
                                          trouser_length=.42,hem_length=.4,skirt_flare=.2,warnings=[]),warnings)
    assert value['garment']['upper']=='shirt'
    assert value['garment']['lower']=='shorts'
    assert 'simplified' in warnings[0]
    assert value['outfit']=='casual'


def test_no_evidence_never_means_naked_or_invented_garment():
    value=appearance.default_appearance()
    warnings=[]
    appearance._garment_from_evidence(value, {}, None, warnings)
    assert value['garment']['upper']==value['garment']['lower']=='unknown'
    assert value['garment']['inferred']
    assert len(warnings)==2
    assert 'unclear' in appearance.appearance_summary(value)


def test_parser_rejects_model_tampering_before_runtime_import(monkeypatch,tmp_path):
    (tmp_path/garment_parser.MODEL_FILE).write_bytes(b'bad model')
    monkeypatch.setattr(garment_parser,'MODEL_DIR',tmp_path)
    with pytest.raises(ValueError,match='checksum mismatch'):
        garment_parser.GarmentRuntime()


@pytest.mark.skipif(not garment_parser.available()[0], reason='Verified local garment parser not installed')
def test_real_parser_does_not_find_clothing_in_blank_image():
    runtime=garment_parser.GarmentRuntime()
    try:
        measured=runtime.parse(Image.new('RGB',(200,400),(240,240,240)),subject())
    finally:
        runtime.close()
    assert measured['torso_clothes'] < .1
    assert not measured['visible_clothes']
