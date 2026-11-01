"""S11's two accepted teacher intervals. NumPy export; official processors on GPU.

No LeRobotDataset statistics are created or passed to policy normalization.
The terminal N+1 state has no fabricated action. Chunk masking follows v0.6.1
SmolVLA's action_is_pad convention (clamp within this episode, mask overflow).
"""
import hashlib,json
from pathlib import Path
import numpy as np

FIELDS={'agent_raw':'env_step_agent_raw','wrist_raw':'env_step_wrist_raw',
        'eef_pos':'env_step_eef_pos','eef_quat_xyzw':'env_step_eef_quat_xyzw',
        'gripper_qpos':'env_step_gripper_qpos','state8':'env_step_robot_state8'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]

def chunk_indices(anchor,length,chunk=50):
    if not 0<=anchor<length:raise IndexError('Anchor outside its episode')
    ids=anchor+np.arange(chunk)
    return np.minimum(ids,length-1),ids>=length

def sources(root):
    a=root/'results/s8_teacher_goal2_init0_live_v1';b=root/'results/s9_teacher_retreat_live_v1';c=root/'results/s10b_clearance_branch_live_v1'
    ia=rows(a/'BOUNDARY_INDEX.jsonl');ib=rows(b/'BOUNDARY_INDEX.jsonl');ic=rows(c/'BOUNDARY_INDEX.jsonl')
    za=np.load(a/'trace.npz');zb=np.load(b/'trace.npz');zc=np.load(c/'trace.npz')
    assert len(zb['actions'])==1032 and len(ib)==161
    for k in za.files:
        stop=872 if k=='actions' else 873
        assert za[k][:stop].tobytes()==zb[k][:stop].tobytes(),('S8/S9 physics join',k)
    pa=np.load(a/ia[872]['file']);pb=np.load(b/ib[0]['file'])
    join={key:{'byte_equal':pa[key].tobytes()==pb[key].tobytes(),'max_abs_difference':float(np.max(np.abs(pa[key].astype(float)-pb[key].astype(float))))} for key in FIELDS.values()}
    assert all(x['byte_equal'] for x in join.values())
    mixed=[(a,x) for x in ia[:872]]+[(b,x) for x in ib]
    # State872's new input is S9's restored boundary; action871's actor proof
    # remains the original S8 post872 (not the restoration's display label).
    posts=[(a,x) for x in ia[:873]]+[(b,x) for x in ib[1:]]
    for t,(_,post) in enumerate(posts[1:]):
        assert post['post_settle_control_step']==t+1
        assert post['actor']=='Gemini'
        expected='Gemini' if t<872 else 'new Gemini continuation, human-specified upward strategy'
        assert post['decision_source']==expected,('Nonteacher S9 row',t)
        assert np.array_equal(post['incoming_raw_action'],zb['actions'][t])
    for t in range(20,1010):
        assert ic[t+1]['incoming_action_teacher_label'] is True
        assert np.array_equal(ic[t+1]['incoming_raw_action'],zc['actions'][t])
    task=read(a/'INITIAL_IDENTITY.json')['instruction']
    assert task==read(c/'INITIAL_IDENTITY.json')['instruction']
    return [dict(id='S9',start=0,end=1032,trace=zb,boundaries=mixed,post_boundaries=posts,task=task),
            dict(id='S10B',start=20,end=1010,trace=zc,boundaries=[(c,x) for x in ic],post_boundaries=[(c,x) for x in ic],task=task)],join

def export(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    streams,join=sources(root);manifest=dict(version='S11_TEACHER_PAIRS_V1',fps=20,total=2022,episodes=[],join872=join,
       alignment='pre[t] official env-return observation -> trace action[t]; actor is post[t+1]',
       observations='raw RGB unchanged; no side/GT/annotations; stored official8D is a diagnostic and recreated from original cached quaternion on GPU',
       timing='env.step cached observations at control boundary; sensor2ms lag is not corrected or asserted fresh',
       independent_initial_states=1,normalizer_stats='none created; original checkpoint processors required',
       quality_adoption={'S9':'Original XY10mm failure unchanged; separate S9B5s stability + parent/user visual adoption. Original S8 observer40 and S9B observer100 excluded.',
                         'S10B':'Preventive human-specified correction; manually chosen checkpoint after original S10 failure. Continuous accepted teacher360+new630. Not proof of rescuing a student failure; original S0 init0 natural continuation was stable.'})
    from PIL import Image,ImageDraw
    for s in streams:
        n=s['end']-s['start'];dest=out/s['id'];dest.mkdir();sample=np.load(s['boundaries'][s['start']][0]/s['boundaries'][s['start']][1]['file'])
        arrays={key:np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=sample[src].dtype,shape=(n,*sample[src].shape)) for key,src in FIELDS.items()}
        origin=[]
        for local,t in enumerate(range(s['start'],s['end'])):
            run,pre=s['boundaries'][t];postrun,post=s['post_boundaries'][t+1];path=run/pre['file']
            assert sha(path)==pre['sha256']
            data=np.load(path)
            for key,src in FIELDS.items():arrays[key][local]=data[src]
            if s['id']=='S9':label='S8_original_teacher' if t<872 else 'S9_Gemini_retreat'
            else:label='S10_original_teacher' if t<380 else 'S10B_new_Gemini'
            origin.append(dict(row=local,source_step=t,source_pre=path.relative_to(root).as_posix(),source_pre_sha256=pre['sha256'],
                               source_post_frame=post['frame_id'],actor_from_post=post.get('incoming_action_source',post.get('decision_source')),teacher_origin=label,
                               source_sim_time_s=pre['sim_time_s'],timestamp_s=local/20))
            if local in {0,n-1, min(50,n-1)}:
                im=Image.new('RGB',(720,408),(14,20,26));d=ImageDraw.Draw(im)
                d.text((8,6),f"S11 {s['id']} pre source{t} -> action{t} / actor from post{t+1}",fill='white')
                for j,key in enumerate(['agent_raw','wrist_raw']):im.paste(Image.fromarray(np.ascontiguousarray(arrays[key][local][::-1])),(360*j,28))
                d.text((8,390),'Viewing flip only; exported training RGB remains unchanged',fill='white');im.save(dest/f'PREVIEW_ROW_{local:04d}.png')
        for a in arrays.values():a.flush()
        np.save(dest/'actions.npy',s['trace']['actions'][s['start']:s['end']])
        (dest/'ORIGINS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in origin))
        info=dict(id=s['id'],rows=n,task=s['task'],source_action_range=[s['start'],s['end']],
                  teacher_origin_counts={x:sum(o['teacher_origin']==x for o in origin) for x in sorted({o['teacher_origin'] for o in origin})},
                  files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.iterdir()) if p.is_file()},
                  terminal_obs_without_action='not exported as a training row',chunk_size=50,tail_padded_anchors=49)
        manifest['episodes'].append(info)
    assert sum(x['rows'] for x in manifest['episodes'])==2022
    (out/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest

class TeacherData:
    """Memory-mapped uint8 originals; no model/normalizer construction in this class."""
    def __init__(self,path):
        self.path=Path(path);self.meta=read(self.path/'MANIFEST.json');self.episodes=[]
        for e in self.meta['episodes']:
            p=self.path/e['id'];self.episodes.append({k:np.load(p/(k+'.npy'),mmap_mode='r') for k in [*FIELDS,'actions']})
        self.lengths=[e['rows'] for e in self.meta['episodes']];self.total=sum(self.lengths)
    def locate(self,anchor):
        if not 0<=anchor<self.total:raise IndexError('Invalid global anchor')
        return (0,anchor) if anchor<self.lengths[0] else (1,anchor-self.lengths[0])
    def batch(self,anchors,env_pre,pre):
        import torch
        from lerobot.envs.utils import preprocess_observation
        loc=[self.locate(int(x)) for x in anchors]
        get=lambda k:np.stack([self.episodes[e][k][i] for e,i in loc])
        obs={'pixels':{'image':get('agent_raw'),'image2':get('wrist_raw')},
             'robot_state':{'eef':{'pos':get('eef_pos'),'quat':get('eef_quat_xyzw')},'gripper':{'qpos':get('gripper_qpos')}}}
        raw=preprocess_observation(obs);raw['task']=[self.meta['episodes'][e]['task'] for e,i in loc]
        raw=env_pre(raw)
        expected=torch.from_numpy(get('state8').astype(np.float32))
        if not torch.allclose(raw['observation.state'].cpu(),expected,atol=1e-6,rtol=0):raise RuntimeError('Official raw8D mismatch; do not use fresh controlled-site state')
        actions=[];masks=[]
        for e,i in loc:
            ids,pad=chunk_indices(i,self.lengths[e]);actions.append(self.episodes[e]['actions'][ids]);masks.append(pad)
        raw['action']=torch.from_numpy(np.stack(actions).astype(np.float32))
        result=pre(raw)
        result['action_is_pad']=torch.from_numpy(np.stack(masks)).to(result['action'].device)
        return result,loc
