"""S15 only: pinned S0 -> discarded speed smoke OR 500-update/final-only SFT.

Runs on the approved GPU only. Does not call LeRobot's standard training CLI,
construct dataset statistics, change normalizers, create pods or evaluate policy.
"""
import argparse,copy,hashlib,importlib.metadata,inspect,json,math,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award.sft_data_s11 import TeacherData,sha,read

PROCESSORS=['policy_preprocessor.json','policy_postprocessor.json',
 'policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']

def learning_rate(update):
    if not 1<=update<=500:raise ValueError('Fixed500 optimizer updates')
    return 2e-5*update/50 if update<=50 else 2e-6+(2e-5-2e-6)*.5*(1+math.cos(math.pi*(update-50)/450))

def tensor_hashes(model,frozen):
    import torch
    return {n:hashlib.sha256(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters() if (not p.requires_grad)==frozen}

def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def parity_fixture_batch(value):
    # Saved fixtures contain the original RGB/EEF pose/gripper qpos, but omit
    # auxiliary controller matrix/joint velocities. Official preprocessing keeps
    # None as None; do not turn missing diagnostics into object arrays or zeros.
    if value is None:return None
    if isinstance(value,dict):return {k:parity_fixture_batch(v) for k,v in value.items()}
    return np.asarray(value)[None,...]

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True)
    p.add_argument('--microbatch',type=int,choices=[1,2,4]);p.add_argument('--time-budget-s',type=int,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated project output required')
    if a.time_budget_s<60:raise ValueError('External remaining wall bound required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+a.time_budget_s
    cfg=read(ROOT/'configs/s15_single_demo_sft_v1.json');acq=read(ROOT/'configs/s2_acquisition_manifest.json')
    if a.policy_snapshot.name!=acq['policy']['revision'] or a.backbone_snapshot.name!=acq['backbone']['revision']:raise ValueError('Pinned S0/backbone required; no resume')
    for f in acq['policy']['files']:
        if f['file'] in PROCESSORS and sha(a.policy_snapshot/f['file'])!=f['sha256']:raise RuntimeError('Original processor hash mismatch')
    import torch
    from lerobot.envs.configs import EnvConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    if not torch.cuda.is_available():raise RuntimeError('GPU-only authorized path; no CPU training fallback')
    spec=read(ROOT/'configs/s8_cabinet_comparison_v1.json')
    actual={k:importlib.metadata.version(k) for k in spec['runtime']}
    if actual!=spec['runtime']:raise RuntimeError('Pinned runtime mismatch')
    refs=read(ROOT/'docs/reference_sources/s11_lerobot_v061/SOURCE_MANIFEST.json')['files'];source_checks={}
    import lerobot
    package=Path(inspect.getfile(lerobot)).parent
    for rel,info in refs.items():
        f=package/rel.removeprefix('src/lerobot/');source_checks[rel]=sha(f)==info['sha256']
    if not all(source_checks.values()):raise RuntimeError('Pinned official training/runtime source mismatch')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    pcfg=SmolVLAConfig.from_pretrained(a.policy_snapshot)
    pcfg.device='cuda';pcfg.use_amp=False;pcfg.n_action_steps=50
    pcfg.freeze_vision_encoder=True;pcfg.train_expert_only=True;pcfg.train_state_proj=True;pcfg.vlm_model_name=str(a.backbone_snapshot)
    pcfg.optimizer_lr=2e-5;pcfg.optimizer_betas=(.9,.95);pcfg.optimizer_eps=1e-8;pcfg.optimizer_weight_decay=1e-10;pcfg.optimizer_grad_clip_norm=10
    pcfg.scheduler_warmup_steps=50;pcfg.scheduler_decay_steps=500;pcfg.scheduler_decay_lr=2e-6
    if pcfg.chunk_size!=50:raise RuntimeError('Keep original chunk50')
    policy=SmolVLAPolicy.from_pretrained(a.policy_snapshot,config=pcfg,strict=True).to('cuda')
    policy.train()
    allowed=('model.vlm_with_expert.lm_expert.','model.state_proj.','model.action_in_proj.','model.action_out_proj.','model.action_time_mlp_in.','model.action_time_mlp_out.')
    names={n:{'requires_grad':p.requires_grad,'numel':p.numel(),'shape':list(p.shape)} for n,p in policy.named_parameters()}
    trainable=[p for n,p in policy.named_parameters() if p.requires_grad]
    if any(p.requires_grad and not n.startswith(allowed) for n,p in policy.named_parameters()):raise RuntimeError('Unexpected trainable backbone/parameter')
    if any(not any(n.startswith(prefix) and v['requires_grad'] for n,v in names.items()) for prefix in allowed):raise RuntimeError('Missing expert/projection update group')
    frozen_before=tensor_hashes(policy,True)
    trainable_before=tensor_hashes(policy,False)
    pre,post=make_pre_post_processors(policy_cfg=pcfg,pretrained_path=str(a.policy_snapshot),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':spec['input']['rename_map']}})
    env_cfg=EnvConfig.get_choice_class('libero')(task='libero_goal',observation_height=360,observation_width=360)
    env_pre,_=make_env_pre_post_processors(env_cfg=env_cfg,policy_cfg=pcfg)
    data=TeacherData(a.data)
    if data.total!=1028 or data.lengths!=[1028]:raise RuntimeError('Exactly one adopted1028 teacher interval required')
    save_json(out/'REQUIRES_GRAD.json',{'flags':{k:getattr(pcfg,k) for k in ['freeze_vision_encoder','train_expert_only','train_state_proj','use_amp']},'parameters':names,'trainable_numel':sum(p.numel() for p in trainable),'total_numel':sum(p.numel() for p in policy.parameters()),'source_checks':source_checks})
    # Actual env formatter -> online preprocessing versus minimal exported raw fields.
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from types import SimpleNamespace
    probes=[];fixtures=read(ROOT/'fixtures/s15_processor_inputs/PROVENANCE.json')
    for anchor in [0,128,350,799,906,943,963,1027]:
        e,i=data.locate(anchor);sample=data.episodes[e]
        reference=fixtures[str(anchor)];original_path=ROOT/reference['fixture']
        if sha(original_path)!=reference['original_sha256']:raise RuntimeError('Original-boundary parity fixture changed')
        original=np.load(original_path)
        formatter=LiberoEnv.__new__(LiberoEnv);formatter._env=SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace(ee_ori_mat=None))])
        formatter.camera_name=['agentview_image','robot0_eye_in_hand_image'];formatter.camera_name_mapping={'agentview_image':'image','robot0_eye_in_hand_image':'image2'};formatter.obs_type='pixels_agent_pos'
        raw={'agentview_image':original['env_step_agent_raw'],'robot0_eye_in_hand_image':original['env_step_wrist_raw'],'robot0_eef_pos':original['env_step_eef_pos'],'robot0_eef_quat':original['env_step_eef_quat_xyzw'],'robot0_gripper_qpos':original['env_step_gripper_qpos']}
        online=preprocess_observation(parity_fixture_batch(formatter._format_raw_obs(raw)));online['task']=[data.meta['episodes'][e]['task']]
        online=pre(env_pre(online)) # exact original MeasurementStack order
        learned,_=data.batch([anchor],env_pre,pre)
        keys=[k for k in online if k.startswith('observation.') and isinstance(online[k],torch.Tensor)]
        diffs={k:float((online[k].float()-learned[k].float()).abs().max()) for k in keys}
        if any(v!=0 for v in diffs.values()):raise RuntimeError('Online/training processor mismatch')
        ids=np.minimum(i+np.arange(50),data.lengths[e]-1);target=torch.from_numpy(np.array(sample['actions'][ids],dtype=np.float32)).to('cuda')
        restored=post(learned['action'])[0].to('cuda');err=float((restored-target).abs().max())
        if err>2e-6:raise RuntimeError('Frozen action normalizer roundtrip failed')
        probes.append(dict(anchor=anchor,original_boundary_fixture_sha256=sha(original_path),online_training_max_abs=diffs,action_roundtrip_max_abs=err))
    save_json(out/'PROCESSOR_PROBE.json',{'probes':probes,'dataset_stats_passed':False,'original_processor_files':{n:sha(a.policy_snapshot/n) for n in PROCESSORS}})

    def optimizer():return torch.optim.AdamW(trainable,lr=2e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    opt=optimizer();rng=np.random.default_rng(42);origin_counts={};samples=[]
    def gradient_groups():
        values={prefix:float(torch.stack([p.grad.detach().float().square().sum() for n,p in policy.named_parameters()
                    if n.startswith(prefix) and p.grad is not None]).sum().sqrt()) for prefix in allowed}
        if any(not math.isfinite(v) or v<=0 for v in values.values()):raise RuntimeError('Expert/projection gradient missing or nonfinite')
        return values
    def update(anchors,micro,lr):
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=lr
        valid_total=sum(min(50,data.lengths[e]-i) for e,i in map(data.locate,anchors))
        total=0.
        for j in range(0,len(anchors),micro):
            b,loc=data.batch(anchors[j:j+micro],env_pre,pre);loss,_=policy(b)
            if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite loss')
            # Exactly the official full-effective-batch masked mean, including
            # variable tail padding; not an average of unequal micro means.
            weight=float((~b['action_is_pad']).sum())/valid_total
            (loss*weight).backward();total+=float(loss.detach())*weight
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,10.,error_if_nonfinite=True))
        if any(p.grad is not None for n,p in policy.named_parameters() if not p.requires_grad):raise RuntimeError('Frozen backbone received parameter gradient')
        opt.step();return total,norm

    if a.mode=='smoke':
        profiles=[]
        for micro in [4]:
            if profiles:
                opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                base=torch.cuda.memory_allocated();first=profiles[0]
                peak=max(first['peak_allocated_bytes'],first['peak_reserved_bytes'])
                predicted=base+max(0,peak-base)*micro
                if predicted>.8*torch.cuda.get_device_properties(0).total_memory:
                    profiles.append(dict(microbatch=micro,status='NOT_ATTEMPTED_MEMORY_HEADROOM',predicted_bytes=predicted));break
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();elapsed=[];status='OK'
            try:
                for u in range(25):
                    if time.monotonic()>deadline-40:raise TimeoutError('Smoke remaining wall bound')
                    anchors=rng.integers(0,data.total,size=micro).tolist();torch.cuda.synchronize();t=time.monotonic();loss,norm=update(anchors,micro,2e-5);torch.cuda.synchronize()
                    if u>=5:elapsed.append(time.monotonic()-t)
                if not norm>0:raise RuntimeError('No trainable gradient')
                grads=gradient_groups()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                save_json(out/'SMOKE_RESULT.json',dict(status='OOM_STOP_NO_TRAIN',failed_microbatch=micro,completed_profiles=profiles,weights_not_saved=True))
                raise
            profiles.append(dict(microbatch=micro,status=status,timed_updates=len(elapsed),seconds_per_micro_update=float(np.median(elapsed)) if elapsed else None,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gradient_l2_by_group=grads if status=='OK' else None))
        if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen backbone changed during smoke')
        ok=[x for x in profiles if x['status']=='OK' and x['timed_updates']==20]
        if not ok:raise RuntimeError('No viable smoke profile')
        selected=min(ok,key=lambda x:16/x['microbatch']*x['seconds_per_micro_update'])
        save_json(out/'SMOKE_RESULT.json',dict(status='SMOKE_DISCARDED',profiles=profiles,selected_microbatch=selected['microbatch'],gradient_accumulation=16//selected['microbatch'],conservative_500_update_estimate_s=500*16/selected['microbatch']*selected['seconds_per_micro_update'],weights_not_saved=True,frozen_unchanged=True,elapsed_s=time.monotonic()-start))
        return 0

    if a.microbatch is None:raise ValueError('Fix microbatch from smoke before training')
    # This process always loaded original S0 above; discarded smoke weights cannot enter it.
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=np.random.default_rng(42)
    for u in range(1,501):
        if time.monotonic()>=deadline-90:raise TimeoutError('Stop without final checkpoint: training deadline')
        anchors=rng.integers(0,data.total,size=16).tolist();samples.append(anchors)
        for anchor in anchors:
            e,i=data.locate(anchor)
            origin='S8_saved_Gemini' if i<730 else 'S14C_saved_Gemini' if i<800 else 'S14D_new_Gemini'
            origin_counts[origin]=origin_counts.get(origin,0)+1
        loss,norm=update(anchors,a.microbatch,learning_rate(u))
        with (out/'UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=u,lr=learning_rate(u),loss=loss,grad_norm=norm,elapsed_s=time.monotonic()-start))+'\n')
    if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen backbone changed')
    trainable_after=tensor_hashes(policy,False);changed=[n for n in trainable_before if trainable_before[n]!=trainable_after[n]]
    if not changed:raise RuntimeError('500 updates did not change trainable parameters')
    save_json(out/'PARAMETER_UPDATE_PROOF.json',dict(changed_trainable_names=changed,before=trainable_before,after=trainable_after,frozen_parameter_hashes=frozen_before))
    final=out/'final';policy.eval();policy.save_pretrained(final)
    for name in PROCESSORS:shutil.copyfile(a.policy_snapshot/name,final/name)
    if any(sha(final/n)!=sha(a.policy_snapshot/n) for n in PROCESSORS):raise RuntimeError('Processor files changed')
    np.save(out/'SAMPLED_ANCHORS.npy',np.asarray(samples,dtype=np.int32))
    # Final save/reload parity; no action selection/queue is needed to compare training loss.
    test,_=data.batch([0,1027],env_pre,pre);noise=torch.randn((2,50,32),device='cuda');fm_time=torch.full((2,),.5,device='cuda')
    with torch.no_grad():before=policy(copy.deepcopy(test),noise=noise,time=fm_time)[0].detach().cpu()
    del opt,trainable,policy;torch.cuda.empty_cache()
    restored_policy=SmolVLAPolicy.from_pretrained(final,config=pcfg,strict=True).to('cuda').eval()
    rpre,rpost=make_pre_post_processors(policy_cfg=pcfg,pretrained_path=str(final),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':spec['input']['rename_map']}})
    test2,_=data.batch([0,1027],env_pre,rpre)
    for key in test:
        if isinstance(test[key],torch.Tensor) and not torch.equal(test[key],test2[key]):raise RuntimeError('Reloaded processor output differs')
    with torch.no_grad():after=restored_policy(copy.deepcopy(test2),noise=noise,time=fm_time)[0].detach().cpu()
    if not torch.allclose(before,after,rtol=0,atol=1e-6):raise RuntimeError('Final checkpoint reload mismatch')
    save_json(out/'TRAIN_RESULT.json',dict(status='TRAINED_500_FINAL_ONLY',updates=500,effective_batch=16,microbatch=a.microbatch,accumulation=16//a.microbatch,source_anchor_counts=origin_counts,sampled_anchors_sha256=sha(out/'SAMPLED_ANCHORS.npy'),dataset_manifest_sha256=sha(a.data/'MANIFEST.json'),training_config_sha256=sha(ROOT/'configs/s15_single_demo_sft_v1.json'),frozen_backbone_unchanged=True,original_processor_files={n:sha(final/n) for n in PROCESSORS},normalizer_stats_overwritten=False,reload_loss_max_abs=float((before-after).abs().max()),final_files={f.name:sha(f) for f in final.iterdir() if f.is_file()},elapsed_s=time.monotonic()-start,new_API_calls=0))
    return 0

if __name__=='__main__':raise SystemExit(main())
