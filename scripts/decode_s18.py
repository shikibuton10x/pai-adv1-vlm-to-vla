"""GPU only on operator GO: fixed-noise teacher-state chunks; no physics/learning."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award.precision_s18 import read,save,common_inference,chunk_metrics,validate_final
from course_award.continue_s17 import validate_source
from course_award.sft_data_s11 import TeacherData,chunk_indices

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);p.add_argument('--model',action='append',required=True);p.add_argument('--time-budget-s',type=int,required=True);a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+a.time_budget_s
 import torch
 from lerobot.envs.configs import EnvConfig
 from lerobot.envs import make_env_pre_post_processors
 from lerobot.policies import make_pre_post_processors
 from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
 from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
 cfg=read(ROOT/'configs/s18_precision_comparison_v1.json');spec=read(ROOT/'configs/s8_cabinet_comparison_v1.json');data=TeacherData(a.data);models=dict(x.split('=',1) for x in a.model)
 if list(models)!=['S0','S15','A','B'] or data.total!=1028:raise ValueError('Fixed4models and1028data')
 results={};all_noise=[]
 for name,path in models.items():
  if time.monotonic()>deadline-40:raise TimeoutError('Decoded wall limit')
  path=Path(path);validate_source(path,cfg) if name=='S15' else validate_final(path,name)
  pc=SmolVLAConfig.from_pretrained(path);pc.device='cuda';pc.use_amp=False;pc.n_action_steps=50;pc.vlm_model_name=str(a.backbone_snapshot)
  policy=SmolVLAPolicy.from_pretrained(path,config=pc,strict=True).to('cuda').eval();dtypes=common_inference(policy,path);save(out/f'{name}_INFERENCE_DTYPES.json',dtypes)
  pre,post=make_pre_post_processors(policy_cfg=pc,pretrained_path=str(path),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':spec['input']['rename_map']}})
  ec=EnvConfig.get_choice_class('libero')(task='libero_goal',observation_height=360,observation_width=360);env_pre,_=make_env_pre_post_processors(env_cfg=ec,policy_cfg=pc)
  predictions=[];normalized=[];targets=[];masks=[];records=[]
  for anchor in cfg['decoded']['anchors']:
   b,_=data.batch([anchor],env_pre,pre);ids,pad=chunk_indices(anchor,1028);target=np.array(data.episodes[0]['actions'][ids],copy=True);anchor_preds=[];anchor_norm=[]
   for seed in cfg['decoded']['noise_seeds']:
    if time.monotonic()>deadline-30:raise TimeoutError('Decoded wall limit')
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn((1,50,32),device='cuda',generator=gen)
    import hashlib
    nh=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest();all_noise.append(dict(model=name,anchor=anchor,seed=seed,sha256=nh))
    policy.reset()
    with torch.inference_mode():pred=policy.predict_action_chunk(dict(b),noise=noise);raw=post(pred)[0].float().cpu().numpy();norm=pred[0].float().cpu().numpy()
    if not np.isfinite(raw).all():raise RuntimeError('Nonfinite decoded action')
    records.append(dict(anchor=anchor,phase=cfg['decoded']['phase_by_anchor'][str(anchor)],noise_seed=seed,noise_sha256=nh,raw=chunk_metrics(raw,target,pad),normalized=chunk_metrics(norm,b['action'][0].float().cpu().numpy(),pad),raw_postprocessor_outside_unit_elements=int((np.abs(raw[~pad])>1).sum()),no_physical_action_applied=True))
    anchor_preds.append(raw);anchor_norm.append(norm)
   predictions.append(anchor_preds);normalized.append(anchor_norm);targets.append(target);masks.append(pad)
  np.savez_compressed(out/f'{name}_DECODED.npz',raw_predictions=np.array(predictions),normalized_predictions=np.array(normalized),raw_targets=np.array(targets),pad_masks=np.array(masks),anchors=np.array(cfg['decoded']['anchors']),noise_seeds=np.array(cfg['decoded']['noise_seeds']))
  results[name]=records;del policy,pre,post;torch.cuda.empty_cache()
 noise_map={}
 for x in all_noise:
  key=(x['anchor'],x['seed']);noise_map.setdefault(key,set()).add(x['sha256'])
 if any(len(v)!=1 for v in noise_map.values()):raise RuntimeError('Decoded noise mismatch')
 save(out/'DECODED_RESULT.json',dict(status='S18_FIXED_TEACHER_CHUNKS_COMPLETE',models=results,noise_records=all_noise,common_noise_equal=True,new_physics=0,new_training=0,raw_vs_applied='Postprocessor outputs only; no implicit clamp, no env.step. Actual rollout raw actions/ctrl are recorded separately.',normalized_gripper_sign_is_not_physical_grip=True))
 return 0
if __name__=='__main__':raise SystemExit(main())
