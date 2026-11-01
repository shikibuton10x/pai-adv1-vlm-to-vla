"""S16 thin S15 wrapper: Z-only interception, one unchanged policy/physics loop."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import ImageDraw
import run_s15_student as baseline
from course_award.height_only_s16 import HeightAssist,VERSION
from course_award.control import verify_controller
from course_award.recording_s15 import StudentRecorder as ExistingRecorder
from course_award.cabinet_recording_s8 import _Deferred
from course_award.diagnostic_s12 import write


def source_files():
    import ast,importlib.util
    found=set()
    def visit(path):
        path=Path(path)
        if path in found or not path.is_file():return
        found.add(path)
        if path.suffix!='.py':return
        rel=path.relative_to(ROOT);package='.'.join(rel.parts[1:-1]) if rel.parts[0]=='src' else ''
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom):
                module=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else (node.module or '')
                modules=[module,*[module+'.'+a.name for a in node.names]]
            for name in modules:
                if name.startswith('course_award.'):
                    target=ROOT/'src'/Path(*name.split('.'));visit(target.with_suffix('.py'));visit(target/'__init__.py')
                elif (ROOT/'scripts'/f'{name}.py').is_file():visit(ROOT/'scripts'/f'{name}.py')
    for name in ['src/course_award/__init__.py','scripts/run_s16_height_only.py','scripts/prepare_s16.py','tests/test_s16_height_only.py',
       'configs/s16_height_only_v1.json','configs/local_assets_linux_s15.json','configs/s15_init0_diagnostic.json','configs/s15_single_demo_sft_v1.json','configs/s8_cabinet_comparison_v1.json',
       'configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt','scripts/acquire_s2_assets.py','scripts/bootstrap_s3.sh','scripts/bootstrap_s15.sh',
       'docs/S8_G1_REFERENCE_PROVENANCE.json','docs/S16_CARRY_HEIGHT_DIAGNOSTIC_CONTRACT.md','fixtures/s13_collision_boxes.json']:
        visit(ROOT/name)
    for dirname in ['fixtures/s16_prefix500','fixtures/s12_g1']:
        for p in (ROOT/dirname).rglob('*'):
            if p.is_file():found.add(p)
    for name in ['ORIGINAL_INITIAL_IDENTITY.json','INITIAL_SETTLED_STATE.npz','ORIGINAL_BOUNDARY_0000.npz']:found.add(ROOT/'fixtures/s15_train_init0'/name)
    return sorted(found)
def snapshot(out):
    import hashlib
    manifest={}
    for path in source_files():
        dest=Path(out)/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());manifest[str(path.relative_to(ROOT))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(Path(out)/'source_snapshot/MANIFEST.json',manifest);return manifest

def release_order(ep):
    if ep is None:return dict(grounded_before_open=False,reason='No valid episode')
    rows=ep.records;grasp=next((i for i,r in enumerate(rows) if r['lift']['left_fingerpad'] and r['lift']['right_fingerpad']),None)
    release=None if grasp is None else next((i for i in range(grasp+1,len(rows)) if not rows[i]['target']['contacts']),None)
    if release is None:return dict(grounded_before_open=False,reason='No target robot-contact loss after grasp',first_grasp=grasp,first_release=None)
    opening=release
    while opening>0 and rows[opening]['action'] is not None and rows[opening]['action'][6]<0:opening-=1
    start=opening+1 if opening<release else None
    grounded=bool(start is not None and 'wooden_cabinet_1_g9' in rows[start-1]['lift']['support_contacts'])
    return dict(grounded_before_open=grounded,first_grasp=grasp,first_release=release,opening_run_start=start,
        g9_contact_pre_open=False if start is None else 'wooden_cabinet_1_g9' in rows[start-1]['lift']['support_contacts'],
        raw_at_release=rows[release]['action'][6],release_phase=rows[release]['phase'],
        joint_gap_at_release_m=float(np.abs(ep.states[release]['gripper_qpos']).sum()),
        note='Posthoc diagnostic, no auto open. A high drop followed by stability does not satisfy grounded_before_open.')

def main():
    config=json.loads((ROOT/'configs/s16_height_only_v1.json').read_text())
    if (config['start_boundary'],config['target_clearance_m'],config['Z_physical_delta_cap_m'],config['return_center_tolerance_m'],config['max_assisted_actions'],config['end_on_raw_open'],config['policy_cap'],config['n_action_steps'],config['policy_seed'])!=(500,.067,.004,.030,680,False,1180,50,42):raise ValueError('Fixed S16 contract changed')
    holder={};old_build=baseline.adapter.build;old_episode=baseline.StudentEpisode;old_record=baseline.StudentRecorder;old_snapshot=baseline.snapshot_source;old_write=baseline.write_json;old_version=baseline.VERSION
    def build(asset,suite,task,init,out,*,render):
        env,raw,receipt=old_build(asset,suite,task,init,out,render=render)
        controller=verify_controller(env.env)
        write(Path(out)/'S16_CONTROLLER_SCALE.json',dict({k:controller[k] for k in ['input_min','input_max','output_min','output_max','use_delta','use_ori']},Z_only_physical_cap_m=.004,XY_rotation_grip_from_policy_unchanged=True))
        assist=HeightAssist(out,config);holder['assist']=assist
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
            if phase=='student_policy' and pending and pending['Z_override_applied']:phase='student_policy_height_assist'
            row=super().observe(action,phase)
            if assist is not None:assist.after_step(self,row)
            return row
    class Recorder(ExistingRecorder):
        def _caption(self,row):
            super()._caption(row);draw=ImageDraw.Draw(self.last_image);draw.rectangle((0,0,1079,39),fill=(14,20,26))
            incoming=holder['assist'].last_completed;assisted=bool(incoming and incoming['Z_override_applied'])
            phase='FIXED OBSERVER / NOT POLICY' if row['phase'].startswith('common_observer') else 'GT Z DIAGNOSTIC ASSIST' if assisted else 'STUDENT ONLY'
            draw.text((8,3),f'S16 | {phase} | 1x / 20Hz | state {row["step"]}',fill='white',font=self.font)
            draw.text((8,22),'Original policy XY/rotation/grip and queue/RNG | initial10 / compute waits omitted',fill='white',font=self.font)
        def finish(self):
            result=super().finish();result.update(version=VERSION,condition='one GT Z-only assisted diagnostic, not unassisted model improvement')
            write(self.out/'VIDEO_RECEIPT.json',result);return result
        def record(self,env,raw,row,*,gripper,decision_id):
            real,offset=self.writer,self.index.tell();self.writer=_Deferred()
            try:value=super().record(env,raw,row,gripper=gripper,decision_id=decision_id)
            finally:self.writer=real
            context=holder['assist'].last_completed;observer=row['phase'].startswith('common_observer');assisted=bool(not observer and context and context['Z_override_applied'])
            value.update(frame_id=f'S16_{row["step"]:04d}',transition_from_frame_id=None if row['step']==0 else f'S16_{row["step"]-1:04d}',transition_to_frame_id=f'S16_{row["step"]:04d}',condition=VERSION,Z_diagnostic_assist_applied=assisted,incoming_action_student_label=bool(row['step']>0 and not observer and not assisted),possible_training_label=False,
                decision_source='SM + diagnostic GT Z' if assisted else 'fixed observer' if observer else 'SmolVLA_SM_n50',action_provenance=context if not observer else None)
            self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush();self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
            holder['assist'].check_input(row['step'])
            return value
    def writer(path,value):
        if Path(path).name=='RUN_SPEC.json':value=dict(version=VERSION,S16=config,unchanged_S15_scientific_baseline=value)
        old_write(path,value)
    baseline.adapter.build=build;baseline.StudentEpisode=Episode;baseline.StudentRecorder=Recorder;baseline.snapshot_source=snapshot;baseline.write_json=writer;baseline.VERSION=VERSION
    try:code=baseline.main()
    finally:
        baseline.adapter.build=old_build;baseline.StudentEpisode=old_episode;baseline.StudentRecorder=old_record;baseline.snapshot_source=old_snapshot;baseline.write_json=old_write;baseline.VERSION=old_version
        if 'stack_class' in holder:holder['stack_class'].action=holder['original_action']
    assist=holder.get('assist')
    if assist is not None:
        diagnostic=assist.finish();path=assist.out/'RESULT.json';result=json.loads(path.read_text());result.update(condition=VERSION,Z_assist_diagnostic=diagnostic,unassisted_model_improvement_score=False,new_teacher_calls=0,training=False)
        if not diagnostic['prefix500_valid']:result.update(diagnostic_valid=False,full_task_success=False,release_stability_pass=False,teaching_material_candidate=False);code=1
        else:result['diagnostic_valid']=bool(result.get('state_valid'))
        result.update(training_eligible=False,teaching_material_candidate=False,human_quality_accepted=False,diagnostic_GT_assisted=True)
        result['release_order_diagnostic']=release_order(holder['assist'].ep)
        result['assisted_quality_complete']=bool(result.get('full_task_success') and result['release_order_diagnostic']['grounded_before_open'] and diagnostic.get('handover',{}).get('reason')=='CENTRAL_REGION_REACHED')
        write(path,result)
    return code
if __name__=='__main__':raise SystemExit(main())
