"""Replay the immutable v2 action prefix and require exact stored-state identity."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_v2_prefix(run):
    run=Path(run).resolve()
    result=json.loads((run/'RESULT.json').read_text())
    receipt=result['environment']
    if (result['status']!='QUALITY_STOP' or result['recorded_actions']!=163 or
        receipt['suite']!='libero_goal' or receipt['task_id']!=9 or
        receipt['initial_state_id']!=0 or receipt['seed']!=42 or
        receipt['preparation_settle_steps']!=10 or result['whole_history']['clean_so_far']):
        raise ValueError('Expected the ineligible Goal9/init0 v2 prefix of163 actions')
    trace=run/'trace.npz'
    if hashlib.sha256(trace.read_bytes()).hexdigest()!=result['trace_sha256']:
        raise ValueError('Source replay trace SHA mismatch')
    with np.load(trace,allow_pickle=False) as saved:
        arrays={k:saved[k].copy() for k in saved.files}
    actions=arrays['actions']
    if (actions.dtype!=np.float32 or actions.shape!=(163,7) or
        not np.isfinite(actions).all() or np.any(np.abs(actions)>1)):
        raise ValueError('Unsafe source action prefix')
    decisions=[json.loads(x) for x in (run/'decisions.jsonl').read_text().splitlines()]
    images=[json.loads(x) for x in (run/'images.jsonl').read_text().splitlines()]
    return dict(run=str(run),result=result,arrays=arrays,last_motion=decisions[-1]['motion'],
        final_images=images[-1],trace_sha256=result['trace_sha256'])


def compare_replay(source, states, actions, whole, intervention, observation, receipt):
    actual={k:np.stack([row[k] for row in states]) for k in states[0]}
    actual['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
    checks={}
    differences={}
    for key,expected in source['arrays'].items():
        value=actual.get(key)
        same_shape=value is not None and value.shape==expected.shape
        checks[key]=bool(same_shape and np.array_equal(value,expected))
        differences[key]=(float(np.max(np.abs(value-expected))) if same_shape and value.size else 0. if same_shape else None)
    prior=source['result']
    checks['whole_history_exact']=whole==prior['whole_history']
    checks['intervention_history_exact']=intervention==prior['intervention_history']
    checks['old_clean_false_preserved']=not whole['clean_so_far'] and not intervention['clean_so_far']
    checks['source_identity']=all(receipt[k]==prior['environment'][k] for k in
        ['task_name','bddl_sha256','init_sha256','g1_reference_sha256','reset_history'])
    for view in source['final_images']['views']:
        digest=hashlib.sha256(np.asarray(observation[view['source_key']]).tobytes()).hexdigest()
        checks['raw_RGB_'+view['camera']]=digest==view['raw_rgb_sha256']
    return dict(pass_all=all(checks.values()),checks=checks,max_abs_differences=differences,
        gate='exact equality of all recorded prefix arrays, cumulative histories, identities, and raw RGB',
        replayed_actions=len(actions),remaining_physics_steps=290-len(actions),
        source_run=source['run'],source_trace_sha256=source['trace_sha256'])
