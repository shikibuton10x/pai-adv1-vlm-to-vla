"""Offline records/video summary; never creates a simulator or provider client."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to(ROOT):raise ValueError('Run outside this experiment root')
    targets=[run/'ANALYSIS_SUMMARY.json',run/'OBSERVATION_SNAPSHOTS.mp4',run/'INITIAL_FINAL.png',run/'VIDEO_RECEIPT.json']
    if any(p.exists() for p in targets):raise FileExistsError('Preserve existing summary artifacts')
    result=json.loads((run/'RESULT.json').read_text())
    steps,decisions,images=[read_rows(run/name) for name in ['steps.jsonl','decisions.jsonl','images.jsonl']]
    rows=[]
    feedback_matches=[]
    for i,row in enumerate(decisions):
        receipt=row['api']
        request=json.loads((run/'requests'/(receipt['count_reservation']+'_request.json')).read_text())
        payload=json.loads(request['contents'][0]['parts'][0]['text'])
        if i:feedback_matches.append(payload['previous_motion']==decisions[i-1].get('motion'))
        usage=receipt['usage']
        motion=row.get('motion',{})
        input_tokens=usage.get('promptTokenCount',0)
        output_tokens=usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0)
        rows.append(dict(decision=i,operation=row.get('command',{}).get('operation'),
            requested=motion.get('requested'),applied=motion.get('applied'),
            clipped=motion.get('clipped'),steps=motion.get('steps',0),
            actual_delta_world_m=motion.get('actual_delta_world_m'),
            target_error_m=motion.get('target_error_m'),unreached=motion.get('unreached'),
            reason=motion.get('reason',row.get('rejected')),
            input_tokens=input_tokens,visible_tokens=usage.get('candidatesTokenCount',0),
            thinking_tokens_reported=usage.get('thoughtsTokenCount'),total_tokens=usage.get('totalTokenCount'),
            counted_input_tokens=receipt['counted_input_tokens'],
            usage_converted_usd=(input_tokens*2+output_tokens*10)/1e6))
    with np.load(run/'trace.npz',allow_pickle=False) as z:
        eef=z['eef_pos']
        actions=z['actions']
        net=eef[-1]-eef[0]
        path=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum())
        command_peak=float(np.linalg.norm(actions[:,:3]*.05,axis=1).max()) if len(actions) else 0.
    full=steps[-1]['whole']
    lift=steps[-1]['lift']
    per_run_cost=sum(row['usage_converted_usd'] for row in rows)
    manifest=json.loads((run/'source_snapshot/MANIFEST.json').read_text())
    assert all(hashlib.sha256((run/'source_snapshot'/p).read_bytes()).hexdigest()==digest for p,digest in manifest.items())
    assert hashlib.sha256((run/'trace.npz').read_bytes()).hexdigest()==result['trace_sha256']
    summary=dict(status=result['status'],run=run.name,recorded_actions=len(actions),
        physics_including_settle=len(actions)+result['environment']['preparation_settle_steps'],
        sim_after_settle_s=len(actions)/20,worker_wall_s=result['elapsed_wall_s'],
        eef_net_delta_world_m=net.tolist(),eef_net_norm_m=float(np.linalg.norm(net)),eef_path_m=path,
        commanded_translation_norm_peak_m=command_peak,operations=rows,
        previous_motion_feedback_matches=feedback_matches,
        whole_history=full,intervention_history=steps[-1]['intervention'],lift=lift,
        max_target_rise_m=max(row['lift']['rise_m'] for row in steps),
        input_tokens=sum(row['input_tokens'] for row in rows),
        visible_tokens=sum(row['visible_tokens'] for row in rows),
        total_tokens=sum(row['total_tokens'] or 0 for row in rows),
        current_run_usage_converted_usd=per_run_cost,
        common_usage_converted_usd=result['accounting']['committed_micro']/1e6,
        common_generations=result['accounting']['generations'],remaining_generations=20-result['accounting']['generations'],
        pending_ids=result['accounting']['pending_ids'],invoice_checked=False,
        source_snapshot_files_verified=len(manifest),postprocess_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    font=ImageFont.load_default(size=13)
    quality_by_step={row['whole']['step']:row for row in steps}
    def panel(item):
        step=item['control_step']
        frame=Image.new('RGB',(720,448),(14,20,26))
        draw=ImageDraw.Draw(frame)
        draw.text((8,3),'OBSERVATION SNAPSHOTS | 0.8s per boundary | not continuous motion',font=font,fill='white')
        draw.text((8,22),f'{run.name} | control step {step} | simulation time after settle {step/20:.2f}s',font=font,fill='white')
        for col,view in enumerate(item['views']):
            frame.paste(Image.open(view['teacher_file']).convert('RGB'),(col*360,56))
        state=quality_by_step[step]
        text=f"whole clean={state['whole']['clean_so_far']}  lift={state['lift']['lift_confirmed']}"
        if state['whole']['first_contact']:text+=' | robot contact detected'
        draw.text((8,424),text,font=font,fill=(255,110,100) if not state['whole']['clean_so_far'] else (150,235,170))
        return frame
    movie=run/'OBSERVATION_SNAPSHOTS.mp4'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),10,(720,448))
    if not writer.isOpened():raise RuntimeError('Existing mp4v encoder unavailable')
    frames=0
    try:
        for item in images:
            bgr=np.asarray(panel(item))[:,:,::-1]
            for _ in range(8):writer.write(bgr);frames+=1
        bgr=np.asarray(panel(images[-1]))[:,:,::-1]
        for _ in range(16):writer.write(bgr);frames+=1
    finally:writer.release()
    capture=cv2.VideoCapture(str(movie))
    decoded=0
    while True:
        ok,_=capture.read()
        if not ok:break
        decoded+=1
    capture.release()
    if decoded!=frames:raise RuntimeError('Saved snapshot movie frame count mismatch')
    contact=Image.new('RGB',(720,800),(14,20,26))
    draw=ImageDraw.Draw(contact)
    for row,item in enumerate([images[0],images[-1]]):
        step=item['control_step']
        draw.text((8,row*400+8),f'{"Initial" if row==0 else "Final"}: step {step} / whole clean={quality_by_step[step]["whole"]["clean_so_far"]}',font=font,fill='white')
        for col,view in enumerate(item['views']):contact.paste(Image.open(view['teacher_file']),(col*360,row*400+40))
    contact.save(run/'INITIAL_FINAL.png')
    write_json(run/'VIDEO_RECEIPT.json',dict(scope='saved operation-boundary RGB snapshots, not continuous/re-rendered motion',fps=10,frames=frames,decoded_frames=decoded,duration_s=frames/10,sha256=hashlib.sha256(movie.read_bytes()).hexdigest()))
    write_json(run/'ANALYSIS_SUMMARY.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','recorded_actions','eef_net_delta_world_m','eef_net_norm_m','eef_path_m','input_tokens','visible_tokens','total_tokens','current_run_usage_converted_usd','common_usage_converted_usd','remaining_generations']}))


if __name__=='__main__':main()
