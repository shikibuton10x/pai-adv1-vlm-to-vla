"""S21 only: frozen S15 final -> discarded 3-update smoke OR500 FP32 B updates, final only.

Runs on the approved GPU only. Does not call LeRobot's standard training CLI,
construct dataset statistics, change normalizers, create pods or evaluate policy.
"""
import argparse,copy,hashlib,importlib.metadata,inspect,json,math,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award.sft_data_s11 import TeacherData,sha,read
from course_award.continue_s17 import validate_source
from course_award.learn_s21 import sample_anchors,fixed_flow
from course_award.precision_s18 import learning_rate,configure_precision,dtype_hooks,step_metrics

PROCESSORS=['policy_preprocessor.json','policy_postprocessor.json',
 'policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']


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
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','train'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True)
    p.add_argument('--arm',choices=['seed43','release_focus42'],required=True);p.add_argument('--microbatch',type=int,choices=[1,2,4]);p.add_argument('--time-budget-s',type=int,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated project output required')
    if a.time_budget_s<60:raise ValueError('External remaining wall bound required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+a.time_budget_s
    cfg=read(ROOT/'configs/s21_seed_release_v1.json');armcfg=cfg['arms'][a.arm];seed=armcfg['global_seed'];acq=read(ROOT/'configs/s2_acquisition_manifest.json')
    if a.backbone_snapshot.name!=acq['backbone']['revision']:raise ValueError('Pinned backbone required')
    validate_source(a.policy_snapshot,cfg)
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
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    pcfg=SmolVLAConfig.from_pretrained(a.policy_snapshot)
    pcfg.device='cuda';pcfg.use_amp=True;pcfg.n_action_steps=50
    pcfg.freeze_vision_encoder=True;pcfg.train_expert_only=True;pcfg.train_state_proj=True;pcfg.vlm_model_name=str(a.backbone_snapshot)
    pcfg.optimizer_lr=1e-5;pcfg.optimizer_betas=(.9,.95);pcfg.optimizer_eps=1e-8;pcfg.optimizer_weight_decay=1e-10;pcfg.optimizer_grad_clip_norm=10
    pcfg.scheduler_warmup_steps=50;pcfg.scheduler_decay_steps=500;pcfg.scheduler_decay_lr=1e-6
    if pcfg.chunk_size!=50:raise RuntimeError('Keep original chunk50')
    policy=SmolVLAPolicy.from_pretrained(a.policy_snapshot,config=pcfg,strict=True).to('cuda')
    precision=configure_precision(policy,'B');precision.update(experiment_arm=a.arm,training_randomness=armcfg);save_json(out/'PRECISION_IDENTITY.json',precision)
    first_forward,hook_handles,first_outputs=dtype_hooks(policy)
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

    def optimizer():return torch.optim.AdamW(trainable,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    opt=optimizer();rng=np.random.default_rng(armcfg['sampler_seed']);origin_counts={};samples=[];sampling_branches=[]
    def gradient_groups():
        values={prefix:float(torch.stack([p.grad.detach().float().square().sum() for n,p in policy.named_parameters()
                    if n.startswith(prefix) and p.grad is not None]).sum().sqrt()) for prefix in allowed}
        if any(not math.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite expert/projection gradient')
        return values
    def update(anchors,micro,lr,u):
        noise,fm_time,flow_digest=fixed_flow(policy,u,len(anchors),armcfg['flow_seed_base'])
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=lr
        valid_total=sum(min(50,data.lengths[e]-i) for e,i in map(data.locate,anchors))
        total=0.
        for j in range(0,len(anchors),micro):
            b,loc=data.batch(anchors[j:j+micro],env_pre,pre)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss,_=policy(b,noise=noise[j:j+micro],time=fm_time[j:j+micro])
            if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite loss')
            # Exactly the official full-effective-batch masked mean, including
            # variable tail padding; not an average of unequal micro means.
            weight=float((~b['action_is_pad']).sum())/valid_total
            (loss*weight).backward();total+=float(loss.detach())*weight
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,10.,error_if_nonfinite=True))
        if any(p.grad is not None for n,p in policy.named_parameters() if not p.requires_grad):raise RuntimeError('Frozen backbone received parameter gradient')
        opt.step();return total,norm,flow_digest

    if a.mode=='probe':
        timings=[];probe=[]
        for u in range(1,cfg['precision']['probe_updates']+1):
            if time.monotonic()>deadline-40:raise TimeoutError('Probe wall bound')
            before={n:p.detach().clone() for n,p in policy.named_parameters() if p.requires_grad}
            torch.cuda.synchronize();t=time.monotonic()
            loss,norm,digest=update(cfg['precision']['probe_batch'],4,1e-5,u)
            torch.cuda.synchronize();timings.append(time.monotonic()-t)
            stats=step_metrics(policy,before,precision['original_parameters']);del before
            probe.append(dict(update=u,loss=loss,grad_norm=norm,flow_sha256=digest,groups=stats))
        grads_by_group=gradient_groups()
        state_dtypes={n:{k:str(v.dtype) for k,v in opt.state[p].items() if isinstance(v,torch.Tensor)} for n,p in policy.named_parameters() if p.requires_grad}
        if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen backbone changed in probe')
        save_json(out/'FIRST_FORWARD_DTYPES.json',first_forward)
        np.savez_compressed(out/'FIRST_FORWARD_OUTPUTS.npz',**first_outputs)
        save_json(out/'PROBE_RESULT.json',dict(status='S21_SMOKE_DISCARDED',arm=a.arm,gradient_l2_by_group=grads_by_group,updates=cfg['precision']['probe_updates'],records=probe,optimizer_state_dtypes=state_dtypes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds_per_update=float(np.median(timings)),forecast_500_update_s=500*float(np.median(timings)),smoke_weights_not_saved=True,finite=True,update_metrics_are_observed_not_rounding_causal_proof=True,weight_decay=1e-10))
        return 0

    if a.microbatch is None:raise ValueError('Fix microbatch from smoke before training')
    # This fresh process always reloads frozen S15 final; discarded smoke weights cannot enter it.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(armcfg['sampler_seed'])
    for u in range(1,501):
        if time.monotonic()>=deadline-90:raise TimeoutError('Stop without final checkpoint: training deadline')
        anchors,branches=sample_anchors(rng,a.arm);samples.append(anchors);sampling_branches.append(branches)
        for anchor in anchors:
            e,i=data.locate(anchor)
            origin='S8_saved_Gemini' if i<730 else 'S14C_saved_Gemini' if i<800 else 'S14D_new_Gemini'
            origin_counts[origin]=origin_counts.get(origin,0)+1
        loss,norm,flow_digest=update(anchors,a.microbatch,learning_rate(u),u)
        with (out/'UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=u,lr=learning_rate(u),loss=loss,grad_norm=norm,flow_sha256=flow_digest,anchors=anchors,elapsed_s=time.monotonic()-start))+'\n')
    if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen backbone changed')
    trainable_after=tensor_hashes(policy,False);changed=[n for n in trainable_before if trainable_before[n]!=trainable_after[n]]
    save_json(out/'PARAMETER_UPDATE_PROOF.json',dict(changed_trainable_names=changed,before=trainable_before,after=trainable_after,frozen_parameter_hashes=frozen_before))
    save_json(out/'FIRST_FORWARD_DTYPES.json',first_forward)
    np.savez_compressed(out/'FIRST_FORWARD_OUTPUTS.npz',**first_outputs)
    final=out/'final';policy.eval();policy.save_pretrained(final)
    for name in PROCESSORS:shutil.copyfile(a.policy_snapshot/name,final/name)
    if any(sha(final/n)!=sha(a.policy_snapshot/n) for n in PROCESSORS):raise RuntimeError('Processor files changed')
    np.save(out/'SAMPLED_ANCHORS.npy',np.asarray(samples,dtype=np.int32))
    np.save(out/'SAMPLING_BRANCHES.npy',np.asarray(sampling_branches,dtype=np.uint8))
    # Final save/reload parity; no action selection/queue is needed to compare training loss.
    test,_=data.batch([0,1027],env_pre,pre);noise=torch.randn((2,50,32),device='cuda');fm_time=torch.full((2,),.5,device='cuda')
    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):before=policy(copy.deepcopy(test),noise=noise,time=fm_time)[0].detach().cpu()
    del opt,trainable,policy;torch.cuda.empty_cache()
    restored_policy=SmolVLAPolicy(pcfg).to('cuda');configure_precision(restored_policy,'B')
    from safetensors.torch import load_model
    load_model(restored_policy,str(final/'model.safetensors'),strict=True,device='cuda');restored_policy.eval()
    rpre,rpost=make_pre_post_processors(policy_cfg=pcfg,pretrained_path=str(final),preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':spec['input']['rename_map']}})
    test2,_=data.batch([0,1027],env_pre,rpre)
    for key in test:
        if isinstance(test[key],torch.Tensor) and not torch.equal(test[key],test2[key]):raise RuntimeError('Reloaded processor output differs')
    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):after=restored_policy(copy.deepcopy(test2),noise=noise,time=fm_time)[0].detach().cpu()
    if not torch.allclose(before,after,rtol=0,atol=1e-6):raise RuntimeError('Final checkpoint reload mismatch')
    save_json(out/'TRAIN_RESULT.json',dict(status='S21_TRAINED_500_FINAL_ONLY',arm=a.arm,training_randomness=armcfg,precision_arm='B',updates=500,changed_trainable_tensor_count=len(changed),zero_parameter_change_is_valid_observation=True,cumulative_nominal_updates=1000,compute_context='shared explicit CUDA BF16 autocast',source_checkpoint_weight_sha256=cfg['source_checkpoint']['files']['model.safetensors'],optimizer_state_reused=False,sampling=dict(cfg['sampling'],arm=armcfg['sampling']),sampling_branches_sha256=sha(out/'SAMPLING_BRANCHES.npy'),effective_batch=16,microbatch=a.microbatch,accumulation=16//a.microbatch,source_anchor_counts=origin_counts,sampled_anchors_sha256=sha(out/'SAMPLED_ANCHORS.npy'),dataset_manifest_sha256=sha(a.data/'MANIFEST.json'),training_config_sha256=sha(ROOT/'configs/s21_seed_release_v1.json'),frozen_backbone_unchanged=True,original_processor_files={n:sha(final/n) for n in PROCESSORS},normalizer_stats_overwritten=False,reload_loss_max_abs=float((before-after).abs().max()),final_files={f.name:sha(f) for f in final.iterdir() if f.is_file()},elapsed_s=time.monotonic()-start,new_API_calls=0))
    return 0

if __name__=='__main__':raise SystemExit(main())
