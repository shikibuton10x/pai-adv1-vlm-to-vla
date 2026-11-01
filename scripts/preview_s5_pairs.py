"""Eight fresh-process initial-only checks, each <=90sec; API/GPU zero."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.development_baselines import MATRIX,compare_pair
from run_s0 import write_json

def main():
 out=ROOT/'results/s5_initial_pair_preview_v1';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();pairs=[]
 for name,i,n in MATRIX:
  cmd=[sys.executable,'-B',str(ROOT/'scripts/preview_s5_initial.py'),'--init',str(i),'--out',str(out/name)]
  with (out/(name+'.log')).open('xb') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=min(90,480-(time.monotonic()-start)))
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise RuntimeError('Initial-only preview timed out')
  if code:raise RuntimeError('Initial-only preview failed: '+name)
  print(name+' PASS',flush=True)
  if n==50:
   pair=compare_pair(out,i);pairs.append(pair);write_json(out/f'INIT{i}_PAIR_CHECK.json',pair)
   if not pair['passed']:raise RuntimeError('Pair initial state/observation mismatch')
 write_json(out/'PREVIEW_VERIFICATION.json',dict(status='PASS',initial_pairs=pairs,fresh_processes=8,policy_loaded=False,student_actions=0,new_teacher_generations=0,GPU_used=False,wall_s=time.monotonic()-start))
if __name__=='__main__':main()
