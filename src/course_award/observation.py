"""Self-EEF annotation only: no target positions or detections in teacher images."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from .control import pose
from .env import sha256


def project(point, camera_pos, camera_rotation, fovy, height, width):
    relative=camera_rotation.T@(np.asarray(point)-camera_pos)
    depth=-float(relative[2])
    if depth<=0:return None
    focal=.5*height/np.tan(.5*np.deg2rad(fovy))
    return np.array([width/2+focal*relative[0]/depth,
                     height/2-focal*relative[1]/depth])


def observe_images(env, obs, out, sequence):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    d=env.env
    eef,_=pose(d)
    points=[eef,*[eef+np.eye(3)[i]*.05 for i in range(3)]]
    rows=[]
    teacher_paths=[]
    for label,camera,key in [('agent','agentview','agentview_image'),('wrist','robot0_eye_in_hand','robot0_eye_in_hand_image')]:
        raw=np.asarray(obs[key]).copy()
        if raw.shape!=(360,360,3) or raw.dtype!=np.uint8:
            raise ValueError('Camera uint8 RGB360 contract failed')
        natural=np.ascontiguousarray(raw[::-1])
        prefix=out/f'{sequence:03d}_{label}'
        raw_path=prefix.with_name(prefix.name+'_raw.png')
        natural_path=prefix.with_name(prefix.name+'_natural.png')
        teacher_path=prefix.with_name(prefix.name+'_teacher.png')
        Image.fromarray(raw).save(raw_path)
        Image.fromarray(natural).save(natural_path)
        cam=d.sim.model.camera_name2id(camera)
        pixels=[project(p,d.sim.data.cam_xpos[cam],d.sim.data.cam_xmat[cam].reshape(3,3),
            d.sim.model.cam_fovy[cam],360,360) for p in points]
        # Independent check against installed robosuite camera convention.
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        matrix=get_camera_transform_matrix(d.sim,camera,360,360)
        differences=[]
        for point,pixel in zip(points,pixels):
            homogeneous=matrix@np.r_[point,1.]
            if pixel is not None:
                differences.append(float(np.max(np.abs(pixel-homogeneous[:2]/homogeneous[2]))))
        if differences and max(differences)>1e-5:
            raise RuntimeError('Self-EEF projection disagrees with installed camera convention')
        image=Image.fromarray(natural)
        if label=='agent':
            draw=ImageDraw.Draw(image)
            for i in range(1,6):
                x=round(360*i/6)
                draw.line((x,0,x,359),fill=(80,110,140))
                draw.line((0,x,359,x),fill=(80,110,140))
            p=pixels[0]
            if p is not None and 0<=p[0]<360 and 0<=p[1]<360:
                draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill='white')
                draw.text((p[0]+4,p[1]+3),'EEF',fill='white')
                for i,(name,color) in enumerate([('X',(255,60,60)),('Y',(60,255,60)),('Z',(70,140,255))]):
                    q=pixels[i+1]
                    if q is not None:
                        draw.line((*p,*q),fill=color,width=2)
                        draw.text(tuple(q),name,fill=color)
            draw.rectangle((0,0,359,15),fill=(0,0,0))
            draw.text((4,3),'World XYZ axes: each 50mm; natural view',fill='white')
        image.save(teacher_path)
        teacher_paths.append(teacher_path)
        rows.append(dict(camera=camera,source_key=key,raw_file=str(raw_path),
            natural_file=str(natural_path),teacher_file=str(teacher_path),
            raw_sha256=sha256(raw_path),teacher_sha256=sha256(teacher_path),
            raw_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            transform='natural=raw[::-1], teacher=natural+self-EEF overlay on agent only',
            eef_axes_world_m=[p.tolist() for p in points],
            natural_pixels=[p.tolist() if p is not None else None for p in pixels],
            corresponding_raw_pixels=[[p[0],359-p[1]] if p is not None else None for p in pixels],
            projection_reference_max_abs_px=max(differences,default=0.),
            object_coordinate_annotation=False))
    return teacher_paths,rows
