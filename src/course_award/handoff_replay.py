"""API-free saved-action prefix. Monitor baselines survive into student control."""
from pathlib import Path
import json,numpy as np
from .env import Monitor,sha256
from .nontarget_v2 import NonTargetMonitor,Retention
from .lift import LiftMonitor
from .control import pose

def replay_prefix(env,source,on_state=None):
    source=Path(source);result=json.loads((source/'RESULT.json').read_text())
    if not result['primary_success']:raise ValueError('Only an accepted successful prefix')
    if sha256(source/'trace.npz')!=result['trace_sha256']:raise ValueError('Source trace SHA mismatch')
    with np.load(source/'trace.npz',allow_pickle=False) as expected:actions=expected['actions'].copy()
    if actions.dtype!=np.float32 or not np.isfinite(actions).all():raise ValueError('Invalid saved actions')
    if actions.shape!=(204,7) or result['retention']['teacher_actions_at_start']!=194:raise ValueError('Wrong HIGH prefix')
    legacy=Monitor(env);nt=NonTargetMonitor(env);lift=LiftMonitor(env);ret=None;raw=None;records=[]
    def observe(i):
        nonlocal ret
        ls=legacy.observe(None if i==0 else actions[i-1]);ns,ts=nt.observe(None if i==0 else actions[i-1]);lf=lift.observe(i)
        if i==194:ret=Retention(i,float(env.env.sim.data.time),lf,model_timestep_s=float(env.env.sim.model.opt.timestep))
        elif i>194:ret.add(i,float(env.env.sim.data.time),lf)
        row=dict(step=i,legacy=ls,nontarget=ns,target=ts,lift=lf);records.append(row)
        if on_state:on_state(i,row)
        if not ns['nontarget_preserved']:raise RuntimeError('Replay non-target violation; no handoff')
    observe(0)
    for i,action in enumerate(actions,1):raw,*_=env.step(action.copy());observe(i)
    held=ret.result()
    if not held['retention_pass'] or not records[-1]['legacy']['clean_so_far']:raise RuntimeError('Replay hold/quality gate failed; no student')
    if not np.all(actions[194:,6]==actions[193,6]):raise RuntimeError('Prefix changed latest gripper in hold')
    return dict(raw_obs=raw,legacy=legacy,nontarget=nt,lift=lift,retention=held,records=records,
        teacher_steps=194,hold_steps=10,settle_steps=10,consumed=214,remaining=600-214)
