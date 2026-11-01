"""Exactly eight fixed S5 baseline cells, fresh processes; no API or retries."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.receiver_contact_v1 import PROFILE,STRICT_PROFILE
from course_award.development_baselines import load_spec,compare_pair
from run_s0 import write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);a=p.parse_args();spec=load_spec(ROOT/'configs/s5_eight_dev_baselines_v1.json');out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S5 output required')
 out.mkdir(parents=True,exist_ok=False);source_snapshot(out,ROOT/'configs/local_assets_linux.json',extra_files=tuple(ROOT/x for x in ['configs/s5_eight_dev_baselines_v1.json','configs/s4_four_dev_cells_v1.json','docs/S5_G1_REFERENCE_PROVENANCE.json','scripts/bootstrap_s3.sh','scripts/bootstrap_s5.sh','configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt']))
 write_json(out/'RUN_SPEC.json',dict(spec=spec,evaluation_profile=PROFILE,strict_diagnostic_profile=STRICT_PROFILE,new_online_teacher_generations=0,reference_init0_not_reexecuted=True,reference_init0_pooled=False,cell_process_isolation=True,work_deadline_epoch=a.work_deadline_epoch,budget_definition='Selected episode: settle10 + student<=590. Prior init reset/settle preparation counted separately.',native_meaning='Original LIBERO predicate, no release/stability requirement',quality_native_definition='native_success AND S4 primary; protection stop first'))
 rows=[];pairs=[];start=time.monotonic();stopped=None
 for cell in spec['cells']:
  limit=min(spec['per_cell_wall_limit_s'],a.work_deadline_epoch-time.time()-10)
  if limit<=0:stopped='EXTERNAL_WORK_CUTOFF';break
  dest=out/cell['id'];cmd=[sys.executable,'-B',str(ROOT/'scripts/run_s5_cell.py'),'--cell',cell['id'],'--out',str(dest),'--policy-snapshot',str(a.policy_snapshot),'--backbone-snapshot',str(a.backbone_snapshot)];started=time.monotonic();timed=False
  with (out/(cell['id']+'_console.log')).open('xb') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    code=process.returncode;timed=True
  path=dest/'RESULT.json';result=json.loads(path.read_text()) if path.is_file() else dict(status='CELL_TIMEOUT' if timed else 'CELL_PROCESS_ERROR',native_success=None,nontarget_preserved=None,quality_native=None)
  row=dict(cell=cell,returncode=code,timed_out=timed,effective_timeout_s=limit,wall_s=time.monotonic()-started,result=result);rows.append(row);write_json(out/(cell['id']+'_OUTCOME.json'),row)
  print(json.dumps(dict(cell=cell['id'],status=result['status'],native=result.get('native_success'),primary=result.get('nontarget_preserved'),strict=result.get('strict_nontarget_preserved'),student_steps=result.get('student_steps'))),flush=True)
  if code or timed:stopped='CELL_INFRASTRUCTURE_OR_BOUNDARY_ERROR';break
  if cell['n_action_steps']==50:
   pair=compare_pair(out,cell['init']);pairs.append(pair);write_json(out/f'INIT{cell["init"]}_PAIR_CHECK.json',pair)
   if pair['status']=='PAIR_MISMATCH':stopped='INITIAL_PAIR_MISMATCH';break
 write_json(out/'COMPARISON.json',dict(version=spec['version'],fixed_cell_order=spec['cell_order'],cells=rows,attempted_cells=len(rows),not_run=spec['cell_order'][len(rows):],stop_reason=stopped,initial_pairs=pairs,prior_results_pooled=False,evaluation_profile=PROFILE,strict_diagnostic_profile=STRICT_PROFILE,new_teacher_calls=0,online_ai_trials_added=0,scope='Fixed development inits4/5/6/9; policyseed42; no heldout or intervention',wall_s=time.monotonic()-start))
if __name__=='__main__':main()
