"""One human-adopted S14D teacher interval; S11 field/mask/processor path unchanged."""
import json
from pathlib import Path
import numpy as np
from .sft_data_s11 import FIELDS,sha,read,rows,chunk_indices,TeacherData
ROOT=Path(__file__).resolve().parents[2]
ANCHORS=[0,128,350,799,906,943,963,1027]

def source():
    p=ROOT/'results/s14d_approach_center_live_v1';idx=rows(p/'BOUNDARY_INDEX.jsonl');z=np.load(p/'trace.npz',allow_pickle=False)
    if z['actions'].shape!=(1128,7) or len(idx)!=1129:raise RuntimeError('Frozen D1128/1129 required')
    for t in range(1028):
        post=idx[t+1]
        if post['post_settle_control_step']!=t+1 or post['incoming_action_teacher_label'] is not True or not np.array_equal(post['incoming_raw_action'],z['actions'][t]):raise RuntimeError('pre/action/post actor mismatch')
    if idx[1029]['incoming_action_teacher_label'] is not False:raise RuntimeError('Observer must be nonteacher')
    return p,idx,z

def export(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);p,idx,z=source();n=1028;dest=out/'S14D';dest.mkdir();contract=ROOT/'docs/S15_ACCEPTED_DEMO_SFT_CONTRACT.md'
    adoption=dict(version='S15_HUMAN_ADOPTION_V1',accepted=True,accepted_action_range=[0,1028],excluded_observer_range=[1028,1128],human_basis='User: broadly good; deep grasp/upper bottle-hand contact and slight retreat wobble noted, considered usable as teacher. Supervisor contract authorizes data use.',contract_sha256=sha(contract),source_RESULT_sha256=sha(p/'RESULT.json'),source_video_sha256=sha(p/'FULL_TASK_1X_20HZ_H264.mp4'),source_trace_sha256=sha(p/'trace.npz'),source_index_sha256=sha(p/'BOUNDARY_INDEX.jsonl'),original_center20mm_success=False,original_center_error_mm=24.596945860517458,raw_original_flags_not_rewritten=True,contact_and_wobble_not_removed=True,independent_episodes=1)
    (out/'HUMAN_ADOPTION.json').write_text(json.dumps(adoption,indent=2)+'\n')
    with np.load(p/idx[0]['file']) as s:arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=s[v].dtype,shape=(n,*s[v].shape)) for k,v in FIELDS.items()}
    provenance=[];fixtures=ROOT/'fixtures/s15_processor_inputs';fixtures.mkdir(parents=True,exist_ok=True);probes={}
    for t in range(n):
        pre,post=idx[t],idx[t+1];f=p/pre['file']
        if sha(f)!=pre['sha256']:raise RuntimeError('Source boundary changed')
        with np.load(f,allow_pickle=False) as original:
            for k,v in FIELDS.items():arrays[k][t]=original[v]
            if t in ANCHORS:
                target=fixtures/f'ORIGINAL_{t:04d}.npz';np.savez_compressed(target,**{v:original[v] for v in FIELDS.values()})
                probes[str(t)]=dict(fixture=str(target.relative_to(ROOT)),original_sha256=sha(target),source_boundary=str(f.relative_to(ROOT)),source_boundary_sha256=pre['sha256'],source_step=t)
        origin='S8_saved_Gemini' if t<730 else 'S14C_saved_Gemini' if t<800 else 'S14D_new_Gemini'
        provenance.append(dict(row=t,source_step=t,source_pre=str(f.relative_to(ROOT)),source_pre_sha256=pre['sha256'],source_post_frame=post['frame_id'],source_post_sha256=post['sha256'],actor_from_post=post['incoming_action_source'],teacher_origin=origin,source_sim_time_s=pre['sim_time_s'],timestamp_s=t/20))
    for v in arrays.values():v.flush()
    np.save(dest/'actions.npy',z['actions'][:n]);(dest/'ORIGINS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in provenance));(fixtures/'PROVENANCE.json').write_text(json.dumps(probes,indent=2)+'\n')
    task=read(p/'INITIAL_IDENTITY.json')['instruction'];ep=dict(id='S14D',rows=n,task=task,source_action_range=[0,n],teacher_origin_counts={'S8_saved_Gemini':730,'S14C_saved_Gemini':70,'S14D_new_Gemini':228},files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(dest.iterdir()) if f.is_file()},chunk_size=50,tail_padded_anchors=49,terminal_obs_without_action='post1028 not fabricated as training action')
    m=dict(version='S15_SINGLE_ACCEPTED_TEACHER_V1',fps=20,total=n,episodes=[ep],independent_initial_states=1,alignment='pre[t] official env-return2RGB/8D -> raw action[t]; teacher actor from post[t+1]',observations='Original cached raw RGB/EEF quaternion; no fresh control-site8D/side/GT',normalizer_stats='none created; original public checkpoint processors only',quality_adoption=adoption)
    (out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');return m
