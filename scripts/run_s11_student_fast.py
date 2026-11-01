"""S11 technical recorder restart; original student runner/scientific settings unchanged."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_s11_student as baseline
from course_award.recording_s11_fast import FastStudentRecorder

def main():
    reference=None
    if '--slow-prefix-reference' in sys.argv:
        i=sys.argv.index('--slow-prefix-reference');reference=Path(sys.argv[i+1]);del sys.argv[i:i+2]
    holder={};original_snapshot=baseline.snapshot_source
    def snapshot(out,extra=()):return original_snapshot(out,extra=(*extra,Path(__file__),ROOT/'src/course_award/recording_s11_fast.py',ROOT/'configs/s11_recorder_fast_v1.json'))
    def recorder(out,variant):
        value=FastStudentRecorder(out,variant,prefix_reference=reference);holder['recorder']=value;return value
    baseline.snapshot_source=snapshot;baseline.StudentRecorder=recorder
    code=baseline.main();r=holder.get('recorder')
    if r is not None:
        f=r.out/'RESULT.json';v=json.loads(f.read_text());v.update(recorder_guard_profile='S11_PERIODIC_NATIVE_V1',technical_restart_not_independent_trial=reference is not None,old_slow_prefix_reference=None if reference is None else str(reference))
        integrity=bool(v.get('video',{}).get('recorder_integrity_pass'))
        if not integrity:v.update(status='RECORDER_INTEGRITY_ERROR',state_valid=False,full_task_success=False,release_stability_pass=False,teaching_material_candidate=False);code=1
        if reference is not None and not (r.out/'SLOW_PREFIX_EQUIVALENCE.json').exists():v.update(status='PREFIX_EQUIVALENCE_NOT_REACHED_OR_FAILED',state_valid=False,full_task_success=False,release_stability_pass=False,teaching_material_candidate=False);code=1
        f.write_text(json.dumps(v,indent=2)+'\n')
    return code
if __name__=='__main__':raise SystemExit(main())
