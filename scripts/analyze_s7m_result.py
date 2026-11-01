"""Read-only source analysis, derived summaries/stills only; no environment/API."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/s7m_full_task_teacher_live_goal9_init0_v1'


def intervals(steps):
    result=[]
    for step in steps:
        if result and step==result[-1][1]+1:result[-1][1]=step
        else:result.append([step,step])
    return result


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=json.loads((OUT/'RESULT.json').read_text())
    rows=[json.loads(x) for x in (OUT/'steps.jsonl').read_text().splitlines()]
    index=[json.loads(x) for x in (OUT/'BOUNDARY_INDEX.jsonl').read_text().splitlines()]
    with np.load(OUT/'trace.npz',allow_pickle=False) as z:
        trace={k:z[k] for k in z.files}
    assert len(rows)==len(index)==len(trace['actions'])+1==939
    for i,(row,boundary) in enumerate(zip(rows,index)):
        assert row['step']==boundary['post_settle_control_step']==i
        assert boundary['phase']==row['phase'] and boundary['sim_time_s']==float(trace['time'][i])
        if i:assert np.array_equal(np.asarray(boundary['incoming_raw_action'],np.float32),trace['actions'][i-1])
        else:assert boundary['incoming_raw_action'] is None
        p=OUT/boundary['file'];assert sha(p)==boundary['sha256']
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(z['fresh_robot_state8'][:3],trace['eef_pos'][i])
            assert float(z['fresh_sim_time_s'])==float(trace['time'][i])
        assert all(boundary['proof'].values())
    keys=['left_pad','right_pad','other_finger','hand','arm']
    parts={k:[r['step'] for r in rows if r['target']['parts'][k]] for k in keys}
    paired=[r['step'] for r in rows if r['lift']['left_fingerpad'] and r['lift']['right_fingerpad']]
    qualifies=[r['step'] for r in rows if r['lift']['rise_m']>=.020 and r['lift']['left_fingerpad'] and r['lift']['right_fingerpad'] and not r['lift']['support_contacts']]
    moving=next((r['step'] for r in rows if r['lift']['lift_confirmed']),None)
    native=[r['step'] for r in rows if r['native_success']]
    released=[r['step'] for r in rows if r['native_success'] and not r['target']['contacts']]
    protected=[i for i,role in enumerate(rows[0]['nontarget']['roles']) if role!=0]
    names=rows[0]['nontarget']['object_names']
    protection={names[i]:dict(max_translation_mm=max(r['nontarget']['translation_mm'][i] for r in rows),max_tilt_deg=max(r['nontarget']['tilt_deg'][i] for r in rows)) for i in protected}
    decisions=[json.loads(x) for x in (OUT/'decisions.jsonl').read_text().splitlines()]
    operations=[dict(decision=d['decision_id']+1,before=d['control_step_before'],after=d['motion']['total_actions'],operation=d['motion']['requested'],applied=d['motion']['applied'],actual_steps=d['motion']['steps'],reason=d['motion']['reason'],goal_error_m=d['motion'].get('target_error_m'),goal_error_rad=d['motion'].get('target_error_rad')) for d in decisions if 'motion' in d]
    usages=[json.loads(p.read_text()) for p in sorted((OUT/'requests').glob('*_response.json'),key=lambda p:p.stat().st_mtime)]
    totals={key:sum(x['usageMetadata'].get(key,0) for x in usages) for key in ['promptTokenCount','thoughtsTokenCount','candidatesTokenCount','totalTokenCount']}
    assert totals['promptTokenCount']*2+(totals['thoughtsTokenCount']+totals['candidatesTokenCount'])*10==result['phase_accounting']['committed_micro']
    audit=json.loads((OUT/'RELEASE_AUDIT_01.json').read_text())
    milestones=sorted(set([0,moving or 0,max(range(len(rows)),key=lambda i:rows[i]['lift']['rise_m']),native[0] if native else 0,931,937,938]))
    cap=cv2.VideoCapture(str(OUT/'FULL_TASK_1X_20HZ_H264.mp4'));frames=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frames in milestones:assert cv2.imwrite(str(OUT/f'REVIEW_FRAME{frames:04d}.png'),frame)
        frames+=1
    cap.release();assert frames==939
    ready=ROOT/'results/s7m_full_task_ready_v1/source_snapshot/MANIFEST.json';manifest=json.loads(ready.read_text());assert all(sha(ROOT/p)==h for p,h in manifest.items())
    runtime=json.loads((OUT/'source_snapshot/MANIFEST.json').read_text());assert all(v==manifest[k] for k,v in runtime.items())
    summary=dict(status='S7M_RAW_VIDEO_RESULT_VERIFIED',source_actions=938,source_states=939,all939_boundary_files_SHA_and_time_action_state_links=True,
        all_render_state_preservation_guards_pass=True,full_video_decoded_frames=frames,video_sha256=sha(OUT/'FULL_TASK_1X_20HZ_H264.mp4'),
        full_task_success=False,terminal_native=True,S4_primary_all_history=all(r['nontarget']['nontarget_preserved'] for r in rows),
        old_strict_all_history=all(r['nontarget']['strict_contact_diagnostic']['nontarget_preserved'] for r in rows),
        first_legacy_false=next((r['step'] for r in rows if not r['legacy']['clean_so_far']),None),
        target_parts={k:dict(states=len(v),intervals=intervals(v)) for k,v in parts.items()},paired_pad_intervals=intervals(paired),
        current_airborne_paired_20mm_intervals=intervals(qualifies),moving_lift_first=moving,
        max_target_root_rise_mm=max(r['lift']['rise_m'] for r in rows)*1000,
        native_intervals=intervals(native),native_and_target_robot_contact0_intervals=intervals(released),
        protected_objects=protection,receiver_robot_contact_states=sum(bool(r['nontarget']['receiver_contacts']) for r in rows),
        terminal_gripper_command=float(trace['actions'][-1,6]),terminal_support=rows[-1]['lift']['support_contacts'],
        audit=dict(states=len(audit['states']),duration_s=audit['duration_tick_s'],position_drift_mm=audit['max_position_drift_m']*1000,
                   orientation_drift_deg=float(np.degrees(audit['max_orientation_drift_rad'])),pass_=audit['full_task_success']),
        operations=operations,usage_totals=totals,usage_estimate_USD=result['phase_accounting']['committed_micro']/1e6,
        shared_usage_estimate_USD=result['accounting']['committed_micro']/1e6,unknown_reservations=result['accounting']['pending_ids'],
        source44_matches_reviewed=True,runtime_source_subset_matches=True,ledger_sha256=sha(ROOT/'results/S1_API_LEDGER.jsonl'),
        captured_milestones=milestones,no_post_stop_actions=True,invoice_confirmed=False)
    (OUT/'POST_RUN_VERIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','moving_lift_first','paired_pad_intervals','current_airborne_paired_20mm_intervals','target_parts','native_intervals','native_and_target_robot_contact0_intervals','max_target_root_rise_mm','first_legacy_false','audit','usage_totals','captured_milestones','ledger_sha256']}))


if __name__=='__main__':main()
