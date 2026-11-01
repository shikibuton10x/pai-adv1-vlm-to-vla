"""Unsent state163 request from previously matched saved observations; no transport."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from course_award.budget import Ledger,INPUT_CAP,OUTPUT_CAP,INPUT_MICRO,OUTPUT_MICRO
from course_award.env import sha256
from course_award.gemini import build_request,encoded,request_hash
from course_award.contact_diagnostic import environment_contact_flags,DIAGNOSTIC_SYSTEM
from course_award.teacher_memory import load_config,load_reference,seed_history,decorate_payload,MEMORY_SYSTEM
from run_s0 import write_json
from run_s1 import source_snapshot


def read_rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 out=args.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent new output required')
 out.mkdir(parents=True,exist_ok=False)
 ledger=ROOT/'results/S1_API_LEDGER.jsonl';before=sha256(ledger)
 account=Ledger._summary(read_rows(ledger))  # pure, does not open/create/append a Ledger
 config_path,config=load_config(ROOT)
 source=ROOT/'results/s1_live_goal9_init0_v2'
 replay=ROOT/'results/s1_closure_goal9_init0_v1'
 gate=json.loads((replay/'REPLAY_GATE.json').read_text())
 if not gate['pass_all'] or gate['source_trace_sha256']!=json.loads((source/'RESULT.json').read_text())['trace_sha256']:
  raise ValueError('Expected already proven state163 identity')
 original_steps=read_rows(source/'steps.jsonl');source_decisions=read_rows(source/'decisions.jsonl')
 history=seed_history(source_decisions,original_steps)
 reference,reference_info=load_reference(source)
 (out/'reference_image').mkdir();ref_copy=out/'reference_image/agentview_step0.png';shutil.copyfile(reference,ref_copy)
 write_json(out/'REFERENCE_IMAGE.json',dict(original_path=str(reference),copied_path=str(ref_copy),**reference_info))
 current=read_rows(replay/'steps.jsonl')[163]
 current_images=read_rows(replay/'images.jsonl')[0]
 if current_images['control_step']!=163:raise ValueError('Expected saved current images at163')
 paths=[Path(v['teacher_file']) for v in current_images['views']]
 receipt=json.loads((source/'RESULT.json').read_text())['environment']
 payload=dict(instruction=receipt['instruction'],
  current_test='grasp wine bottle; lift>=20mm and hold0.5s; no rack placement yet',
  robot=current['robot'],gripper_command=-1.,control_step=163,remaining_control_steps=127,
  previous_motion=source_decisions[-1]['motion'],previous_rejection=None,frame_sequence=0,
  axis_length_m=.05,images_order=['agentview_self_EEF_axes','wrist_natural'],
  robot_contact=environment_contact_flags(current['raw_robot_contacts']))
 payload=decorate_payload(payload,history,reference_info)
 request=build_request(payload,paths,system_suffix=DIAGNOSTIC_SYSTEM+MEMORY_SYSTEM,reference_image_path=ref_copy)
 write_json(out/'TEACHER_PAYLOAD.json',payload)
 (out/'REQUEST.json').write_bytes(encoded(request))
 manifest=source_snapshot(out,ROOT/'configs/local_assets.json',extra_files=(config_path,))
 worst=INPUT_CAP*INPUT_MICRO+OUTPUT_CAP*OUTPUT_MICRO
 preview=dict(stage='S1_MEMORY_REFERENCE_V1_PREVIEW_ONLY',provider_requests=0,credential_reads=0,
  new_physics_steps=0,config=config,current_control_step=163,remaining_physics=127,
  previous_replay_identity_gate=str(replay/'REPLAY_GATE.json'),old_quality_remains_ineligible=True,
  source_manifest=manifest,source_trace_sha256=gate['source_trace_sha256'],
  canonical_request_sha256=request_hash(request),current_image_sha256=[sha256(p) for p in paths],
  reference_sha256=sha256(ref_copy),image_count=3,
  payload_utf8_bytes=len(json.dumps(payload,sort_keys=True,allow_nan=False).encode()),
  input_tokens=None,input_tokens_note='Not measured; no countTokens call. Live counts the complete identical request before generation.',
  original_generations_used=account['generations'],original_counts_used=account['counts'],
  pending_ids=account['pending_ids'],common_usage_converted_usd=account['committed_micro']/1e6,
  common_generation_cap=32,additional_generation_cap=12,
  maximum_reservation_per_generation_usd=worst/1e6,
  maximum_additional_12_usd=12*worst/1e6,
  common_maximum_after_12_usd=(account['committed_micro']+12*worst)/1e6,
  common_money_cap_usd=1.5,ledger_sha256_before=before,ledger_sha256_after=sha256(ledger),
  teacher_contact_flags='robot to environment only; robot-to-robot self-collision excluded',
  prefix_history_contact='unknown; never filled from quality/lift/target contact truth',
  exploratory_variant='history + reference view + recovery prompt + corrected environment flags; no single-factor causal claim')
 assert before==preview['ledger_sha256_after']
 assert account['generations']==20 and not account['pending_ids']
 assert preview['common_maximum_after_12_usd']<=1.5
 assert 'previous_motion' not in payload and len(payload['recent_operations'])==4
 write_json(out/'PREVIEW_RECEIPT.json',preview)
 print(json.dumps({k:preview[k] for k in ['stage','provider_requests','new_physics_steps','image_count','payload_utf8_bytes','canonical_request_sha256','common_maximum_after_12_usd']}))
if __name__=='__main__':main()
