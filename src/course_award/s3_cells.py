"""Only the four predeclared S3 cells and their queue/stop boundary."""
import json
from pathlib import Path
MATRIX=(('solo_n10',False,10,590),('solo_n50',False,50,590),('prefix204_n10',True,10,386),('prefix204_n50',True,50,386))
def load_spec(path):
 s=json.loads(Path(path).read_text());actual=tuple((c['id'],c['prefix'],c['n_action_steps'],c['student_steps']) for c in s['cells'])
 if actual!=MATRIX or s['cell_order']!=[c[0] for c in MATRIX]:raise ValueError('Fixed4 cell matrix changed')
 if (s['budget_total'],s['settle_steps'],s['task'],s['init'],s['env_seed'],s['policy_seed'])!=(600,10,9,0,42,42):raise ValueError('S3 common contract changed')
 for c in s['cells']:
  expected=(194,10) if c['prefix'] else (0,0)
  if (c['teacher_steps'],c['hold_steps'])!=expected or 10+sum(expected)+c['student_steps']!=600:raise ValueError('S3 budget split changed')
 if s['prior_s2_reference']['pooled'] or s['new_online_teacher_generations']!=0:raise ValueError('S2/online teacher counting changed')
 return s

def verify_queue(policy,n,*,first_action=False):
 if policy.config.n_action_steps!=n:raise RuntimeError('Loaded policy chunk mismatch')
 q=policy._queues['action'];expected=n-1 if first_action else 0
 if q.maxlen!=n or len(q)!=expected:raise RuntimeError('Policy queue capacity/count mismatch')
 return dict(action=dict(length=len(q),capacity=q.maxlen))

def stop_reason(nontarget_preserved,native_success):
 # Legacy quality is recorded, deliberately not an input to the NT stop decision.
 if not nontarget_preserved:return 'NONTARGET_VIOLATION'
 if native_success:return 'NATIVE_SUCCESS'
 return None

def score_fields(native_success,nontarget_preserved):
 return dict(native_success=native_success,nontarget_preserved=nontarget_preserved,
  quality_native=None if native_success is None or nontarget_preserved is None else bool(native_success and nontarget_preserved))
