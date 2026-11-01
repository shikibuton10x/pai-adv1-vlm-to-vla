"""The fixed four S4 cells, each in a new process. No substitutions/retries/API."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.receiver_contact_v1 import load_spec,PROFILE,STRICT_PROFILE
from course_award.env import sha256
from run_s0 import write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);a=p.parse_args()
 spec=load_spec(ROOT/'configs/s4_four_dev_cells_v1.json');out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated comparison output')
 out.mkdir(parents=True,exist_ok=False);source_snapshot(out,ROOT/'configs/local_assets_linux.json',extra_files=tuple(ROOT/x for x in ['configs/s4_four_dev_cells_v1.json','scripts/bootstrap_s3.sh','configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt']))
 reference=ROOT/spec['prior_s2_reference']['path']
 if sha256(reference)!=spec['prior_s2_reference']['sha256']:raise RuntimeError('S2 reference changed')
 reference_s3=ROOT/spec['prior_s3_reference']['path']
 if sha256(reference_s3)!=spec['prior_s3_reference']['sha256']:raise RuntimeError('S3 reference changed')
 write_json(out/'RUN_SPEC.json',dict(spec=spec,reference_s2_included_in_new_results=False,reference_s3_included_in_new_results=False,evaluation_profile=PROFILE,strict_diagnostic_profile=STRICT_PROFILE,online_teacher_trials_added=0,quality_native_definition='native_success AND S4 primary nontarget_preserved; NT stop has priority',native_meaning='LIBERO predicate, no release/stability requirement',cell_process_isolation=True))
 rows=[];start=time.monotonic()
 for cell in spec['cells']:
  cell_out=out/cell['id'];cmd=[sys.executable,'-B',str(ROOT/'scripts/run_s4_cell.py'),'--cell',cell['id'],'--out',str(cell_out),'--policy-snapshot',str(a.policy_snapshot),'--backbone-snapshot',str(a.backbone_snapshot)];started=time.monotonic();timed=False
  with (out/(cell['id']+'_console.log')).open('xb') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=spec['per_cell_wall_limit_s'])
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    code=process.returncode;timed=True
  path=cell_out/'RESULT.json';result=json.loads(path.read_text()) if path.is_file() else dict(status='CELL_TIMEOUT' if timed else 'CELL_PROCESS_ERROR',evaluation_profile=PROFILE,strict_diagnostic_profile=STRICT_PROFILE,strict_nontarget_preserved=None,strict_quality_native=None,native_success=None,nontarget_preserved=None,quality_native=None)
  row=dict(cell=cell,returncode=code,timed_out=timed,wall_s=time.monotonic()-started,result=result);rows.append(row);write_json(out/(cell['id']+'_OUTCOME.json'),row)
  print(json.dumps(dict(cell=cell['id'],status=result['status'],native=result.get('native_success'),NT=result.get('nontarget_preserved'),quality_native=result.get('quality_native'),student_steps=result.get('student_steps'))),flush=True)
 write_json(out/'COMPARISON.json',dict(version=spec['version'],fixed_cell_order=spec['cell_order'],cells=rows,new_cells=len(rows),s2_pooled=False,s3_pooled=False,evaluation_profile=PROFILE,strict_diagnostic_profile=STRICT_PROFILE,new_teacher_calls=0,online_ai_trials_added=0,scope='one dev init / one policy seed; no generalization claim',wall_s=time.monotonic()-start))
if __name__=='__main__':main()
