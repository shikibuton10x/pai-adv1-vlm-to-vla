"""Thin S12 wrapper: one XY-only action interception, no second policy/physics loop."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import ImageDraw
import run_s12_student as baseline
from course_award.xy_only_s13 import XYAssist,VERSION
from course_award.control import verify_controller
from course_award.recording_s12 import StudentRecorder as ExistingRecorder
from course_award.cabinet_recording_s8 import _Deferred
from course_award.diagnostic_s12 import write


def source_files():
    from course_award.diagnostic_s12 import source_files as existing
    return sorted(set(existing(ROOT)+[ROOT/p for p in ['scripts/run_s13_xy_only.py','scripts/prepare_s13.py','src/course_award/xy_only_s13.py','configs/s13_xy_only_v1.json','fixtures/s13_collision_boxes.json','docs/S13_XY_ONLY_DIAGNOSTIC_CONTRACT.md','tests/test_s13_xy_only.py']]+list((ROOT/'fixtures/s13_prefix700').glob('*'))))
def snapshot(out):
    import hashlib
    manifest={}
    for path in source_files():
        dest=Path(out)/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());manifest[str(path.relative_to(ROOT))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(Path(out)/'source_snapshot/MANIFEST.json',manifest);return manifest

def main():
    config=json.loads((ROOT/'configs/s13_xy_only_v1.json').read_text())
    if (config['start_boundary'],config['max_assisted_actions'],config['XY_physical_norm_cap_m'],config['XY_error_tolerance_m'],config['end_on_raw_open'],config['policy_cap'],config['n_action_steps'],config['policy_seed'])!=(700,160,.004,.002,False,1180,50,42):raise ValueError('Fixed S13 contract changed')
    holder={};old_build=baseline.adapter.build;old_episode=baseline.StudentEpisode;old_record=baseline.StudentRecorder;old_snapshot=baseline.snapshot_source;old_write=baseline.write_json;old_version=baseline.VERSION
    def build(asset,suite,task,init,out,*,render):
        env,raw,receipt=old_build(asset,suite,task,init,out,render=render)
        controller=verify_controller(env.env)
        write(Path(out)/'S13_CONTROLLER_SCALE.json',dict({k:controller[k] for k in ['input_min','input_max','output_min','output_max','use_delta','use_ori']},XY_only_norm_cap_m=.004,Z_rotation_grip_from_policy_unchanged=True))
        assist=XYAssist(out,config);holder['assist']=assist
        # Import only after the unchanged env builder has set LIBERO paths/EGL.
        from course_award.vendor.measurement_stack import MeasurementStack
        original=MeasurementStack.action;holder['original_action']=original;holder['stack_class']=MeasurementStack
        def action(stack,*args,**kwargs):
            before=len(stack.policy._queues['action']);predicted=original(stack,*args,**kwargs);after=len(stack.policy._queues['action'])
            if predicted.shape!=(1,7):raise RuntimeError('Original official action batch changed')
            value=predicted.copy();value[0]=assist.apply(predicted[0].copy(),before,after);return value
        MeasurementStack.action=action
        return env,raw,receipt
    class Episode(old_episode):
        def observe(self,action,phase):
            assist=holder.get('assist');pending=None if assist is None else assist.pending
            if phase=='student_policy' and pending and pending['XY_override_applied']:phase='student_policy_xy_assist'
            row=super().observe(action,phase)
            if assist is not None:assist.after_step(self,row)
            return row
    class Recorder(ExistingRecorder):
        def _caption(self,row):
            super()._caption(row);draw=ImageDraw.Draw(self.last_image);draw.rectangle((0,0,1079,39),fill=(14,20,26))
            incoming=holder['assist'].last_completed;assisted=bool(incoming and incoming['XY_override_applied'])
            phase='FIXED OBSERVER / NOT POLICY' if row['phase'].startswith('common_observer') else 'GT XY DIAGNOSTIC ASSIST' if assisted else 'STUDENT ONLY'
            draw.text((8,3),f'S13 | {phase} | 1x / 20Hz | state {row["step"]}',fill='white',font=self.font)
            draw.text((8,22),'Original policy Z/rotation/grip and queue/RNG | initial10 / compute waits omitted',fill='white',font=self.font)
        def finish(self):
            result=super().finish();result.update(version=VERSION,condition='one GT XY-only assisted diagnostic, not unassisted model improvement')
            write(self.out/'VIDEO_RECEIPT.json',result);return result
        def record(self,env,raw,row,*,gripper,decision_id):
            real,offset=self.writer,self.index.tell();self.writer=_Deferred()
            try:value=super().record(env,raw,row,gripper=gripper,decision_id=decision_id)
            finally:self.writer=real
            context=holder['assist'].last_completed;observer=row['phase'].startswith('common_observer');assisted=bool(not observer and context and context['XY_override_applied'])
            value.update(condition=VERSION,XY_diagnostic_assist_applied=assisted,incoming_action_student_label=bool(row['step']>0 and not observer and not assisted),possible_training_label=False,
                decision_source='SM + diagnostic GT XY' if assisted else 'fixed observer' if observer else 'SmolVLA_SM_n50',action_provenance=context if not observer else None)
            self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush();self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
            holder['assist'].check_input(row['step'])
            return value
    def writer(path,value):
        if Path(path).name=='RUN_SPEC.json':value=dict(version=VERSION,S13=config,unchanged_S12_scientific_baseline=value)
        old_write(path,value)
    baseline.adapter.build=build;baseline.StudentEpisode=Episode;baseline.StudentRecorder=Recorder;baseline.snapshot_source=snapshot;baseline.write_json=writer;baseline.VERSION=VERSION
    try:code=baseline.main()
    finally:
        baseline.adapter.build=old_build;baseline.StudentEpisode=old_episode;baseline.StudentRecorder=old_record;baseline.snapshot_source=old_snapshot;baseline.write_json=old_write;baseline.VERSION=old_version
        if 'stack_class' in holder:holder['stack_class'].action=holder['original_action']
    assist=holder.get('assist')
    if assist is not None:
        diagnostic=assist.finish();path=assist.out/'RESULT.json';result=json.loads(path.read_text());result.update(condition=VERSION,XY_assist_diagnostic=diagnostic,unassisted_model_improvement_score=False,new_teacher_calls=0,training=False)
        if not diagnostic['prefix700_valid']:result.update(diagnostic_valid=False,full_task_success=False,release_stability_pass=False,teaching_material_candidate=False);code=1
        else:result['diagnostic_valid']=bool(result.get('state_valid'))
        write(path,result)
    return code
if __name__=='__main__':raise SystemExit(main())
