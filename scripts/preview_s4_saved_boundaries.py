"""CPU-only shadow evaluation of the four already-truncated S3 traces.
Never changes their labels and never predicts continuation/native success.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.receiver_contact_v1 import ReceiverProtection,TARGET,RECEIVER,PROFILE
from course_award.env import sha256

def main():
 source=ROOT/'results/s3_four_dev_cells_goal9_init0_v1';out=ROOT/'results/s4_receiver_shadow_preview_v1';out.mkdir(exist_ok=False);rows=[]
 for name in ['solo_n10','solo_n50','prefix204_n10','prefix204_n50']:
  p=source/name;result_sha=sha256(p/'RESULT.json');students=[json.loads(x) for x in (p/'student_steps.jsonl').read_text().splitlines()];gate=json.loads((p/'TEACHER_HOLD_GATE.json').read_text());names=students[-1]['nontarget']['object_names'];offset=204 if gate['applicable'] else 0
  contacts={i+offset+1:r['nontarget']['contacts'] for i,r in enumerate(students)}
  if gate['applicable']:contacts.update({r['step']:r['nontarget']['contacts'] for r in gate['records']})
  else:contacts[0]=gate['initial_nontarget']['contacts']
  with np.load(p/'trace.npz',allow_pickle=False) as z:
   g=ReceiverProtection(names,z['object_pos'][0],z['object_quat'][0],target=TARGET,receiver=RECEIVER)
   for i in range(len(z['time'])):value=g.observe(z['object_pos'][i],z['object_quat'][i],contacts[i],i)
  assert value['strict_contact_diagnostic']['nontarget_preserved']==students[-1]['nontarget']['nontarget_preserved'];assert sha256(p/'RESULT.json')==result_sha
  rows.append(dict(cell=name,source_result_sha256=result_sha,source_trace_sha256=sha256(p/'trace.npz'),old_strict_endpoint_preserved=students[-1]['nontarget']['nontarget_preserved'],s4_primary_on_saved_truncated_trace=value['nontarget_preserved'],s4_first_violation=value['first_violation'],s4_first_stopping_contact=value['first_contact'],receiver_contact_states=value['receiver_contact_states'],source_native=False,new_episode=False))
 expected=[True,True,False,False];assert [r['s4_primary_on_saved_truncated_trace'] for r in rows]==expected
 receipt=dict(status='S4_SAVED_BOUNDARY_PREVIEW_PASS',evaluation_profile=PROFILE,source_scores_unchanged=True,new_api=0,new_rollout=0,new_gpu=0,meaning='Only old truncated endpoints; receiver-only stops no longer violate primary. No measured continuation or new success.',cells=rows)
 (out/'PREVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
