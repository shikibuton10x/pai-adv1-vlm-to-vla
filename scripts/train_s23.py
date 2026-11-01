"""S23 only: S22/2000 raw FP32 weights -> discarded smoke OR additional2000 with pure1000/2000 saves.

Runs on the approved GPU only. Does not call LeRobot's standard training CLI,
construct dataset statistics, change normalizers, create pods or evaluate policy.
"""
import argparse,copy,hashlib,importlib.metadata,inspect,json,math,os,pickle,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award.sft_data_s11 import TeacherData,sha,read
from course_award.learn_s23 import validate_source,sample_anchors,fixed_flow,learning_rate,configure_precision,dtype_hooks,pure_save,CHECKPOINTS
from course_award.precision_s18 import step_metrics

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
    p.add_argument('--arm',choices=['full_uniform'],required=True);p.add_argument('--microbatch',type=int,choices=[1,2,4]);p.add_argument('--time-budget-s',type=int,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated project output required')
    if a.time_budget_s<60:raise ValueError('External remaining wall bound required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+a.time_budget_s
    cfg=read(ROOT/'configs/s23_deadline_v1.json');armcfg=cfg['arms'][a.arm];seed=armcfg['global_seed'];acq=read(ROOT/'configs/s2_acquisition_manifest.json')
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
    pcfg.optimizer_lr=1e-4;pcfg.optimizer_betas=(.9,.95);pcfg.optimizer_eps=1e-8;pcfg.optimizer_weight_decay=1e-10;pcfg.optimizer_grad_clip_norm=10
    pcfg.scheduler_warmup_steps=50;pcfg.scheduler_decay_steps=2000;pcfg.scheduler_decay_lr=2e-6
    if pcfg.chunk_size!=50:raise RuntimeError('Keep original chunk50')
    policy=SmolVLAPolicy(pcfg).to('cuda')
    precision=configure_precision(policy,'B')
    from safetensors.torch import load_model
    load_model(policy,str(a.policy_snapshot/'model.safetensors'),strict=True,device='cuda')
    expected=read(a.policy_snapshot.parent/'CHECKPOINT_PARAMETERS.json')
    if {**tensor_hashes(policy,False),**tensor_hashes(policy,True)}!=expected:raise RuntimeError('S22 FP32 source numeric bytes mismatch')
    from course_award.precision_s18 import numeric_hash
    original=read(a.policy_snapshot.parent/'SOURCE_PRECISION_IDENTITY.json')['original_parameters']
    precision.update(original_parameters=original,source='S22/2000 raw FP32; weights-only fresh AdamW',loaded_trainable_dtypes={n:str(p.dtype) for n,p in policy.named_parameters() if p.requires_grad},source_numeric_bytes_equal=True,initial_trainable_numeric_fp32_sha256={n:numeric_hash(p) for n,p in policy.named_parameters() if p.requires_grad},experiment_arm=a.arm,training_randomness=armcfg)
    save_json(out/'PRECISION_IDENTITY.json',precision)
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

    def optimizer():return torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
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

    def training_state():
        h=hashlib.sha256();steps=[]
        for n,p in policy.named_parameters():
            if not p.requires_grad:continue
            state=opt.state[p];steps.append(float(state['step']))
            for key,value in sorted(state.items()):
                h.update((n+'/'+key).encode())
                if isinstance(value,torch.Tensor):h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                else:h.update(repr(value).encode())
        digest=lambda x:hashlib.sha256(x).hexdigest()
        return dict(train_modes={n:m.training for n,m in policy.named_modules()},
            RNG=dict(python=digest(pickle.dumps(random.getstate())),numpy_global=digest(pickle.dumps(np.random.get_state())),
                sampler=rng.bit_generator.state,torch_cpu=digest(torch.get_rng_state().numpy().tobytes()),
                torch_cuda=[digest(x.cpu().numpy().tobytes()) for x in torch.cuda.get_rng_state_all()]),
            optimizer_tensor_SHA256=h.hexdigest(),optimizer_steps_min=min(steps),optimizer_steps_max=max(steps),
            optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],
            parameter_hashes={**tensor_hashes(policy,False),**tensor_hashes(policy,True)})

    def save_checkpoint(u):
        checkpoint=out/f'checkpoint_{u:04d}';checkpoint.mkdir();final=checkpoint/'final'
        def store():
            policy.save_pretrained(final)
            for name in PROCESSORS:shutil.copyfile(a.policy_snapshot/name,final/name)
        proof=pure_save(store,training_state)
        assert proof['before']['optimizer_steps_min']==u==proof['before']['optimizer_steps_max']
        if any(sha(final/n)!=sha(a.policy_snapshot/n) for n in PROCESSORS):raise RuntimeError('Source S0 processor changed')
        save_json(checkpoint/'PURE_SAVE_PROOF.json',proof)
        save_json(checkpoint/'SOURCE_PRECISION_IDENTITY.json',precision)
        save_json(checkpoint/'CHECKPOINT_PARAMETERS.json',proof['before']['parameter_hashes'])
        save_json(checkpoint/'CHECKPOINT_EXPOSURE.json',dict(update=u,anchor_count=u*16,source_anchor_counts=dict(origin_counts)))

    if a.mode=='probe':
        timings=[];probe=[]
        for u in range(1,cfg['precision']['probe_updates']+1):
            if time.monotonic()>deadline-40:raise TimeoutError('Probe wall bound')
            before={n:p.detach().clone() for n,p in policy.named_parameters() if p.requires_grad}
            torch.cuda.synchronize();t=time.monotonic()
            loss,norm,digest=update(cfg['precision']['probe_batch'],4,1e-4,u)
            torch.cuda.synchronize();timings.append(time.monotonic()-t)
            stats=step_metrics(policy,before,precision['original_parameters']);del before
            probe.append(dict(update=u,loss=loss,grad_norm=norm,flow_sha256=digest,groups=stats))
        grads_by_group=gradient_groups()
        state_dtypes={n:{k:str(v.dtype) for k,v in opt.state[p].items() if isinstance(v,torch.Tensor)} for n,p in policy.named_parameters() if p.requires_grad}
        if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen backbone changed in probe')
        save_json(out/'FIRST_FORWARD_DTYPES.json',first_forward)
        np.savez_compressed(out/'FIRST_FORWARD_OUTPUTS.npz',**first_outputs)
        save_json(out/'PROBE_RESULT.json',dict(status='S23_SOURCE_SMOKE_DISCARDED',arm=a.arm,gradient_l2_by_group=grads_by_group,updates=cfg['precision']['probe_updates'],records=probe,optimizer_state_dtypes=state_dtypes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds_per_update=float(np.median(timings)),forecast_2000_update_s=2000*float(np.median(timings)),smoke_weights_not_saved=True,finite=True,update_metrics_are_observed_not_rounding_causal_proof=True,weight_decay=1e-10))
        return 0

    if a.microbatch is None:raise ValueError('Fix microbatch from smoke before training')
    # This fresh process always reloads S22/2000 raw FP32 weights; discarded smoke weights cannot enter it.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(armcfg['sampler_seed'])
    if time.time()>1793575500:raise TimeoutError('Latest main-training start23:25UTC exceeded; zero main updates')
    for u in range(1,1001):
        if time.monotonic()>=deadline-90:raise TimeoutError('Stop without final checkpoint: training deadline')
        anchors=sample_anchors(rng);branches=[0]*16;samples.append(anchors);sampling_branches.append(branches)
        for anchor in anchors:
            e,i=data.locate(anchor)
            origin='S8_saved_Gemini' if i<730 else 'S14C_saved_Gemini' if i<800 else 'S14D_new_Gemini'
            origin_counts[origin]=origin_counts.get(origin,0)+1
        loss,norm,flow_digest=update(anchors,a.microbatch,learning_rate(u),u)
        with (out/'UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=u,lr=learning_rate(u),loss=loss,grad_norm=norm,flow_sha256=flow_digest,anchors=anchors,elapsed_s=time.monotonic()-start))+'\n')
        if u in CHECKPOINTS:save_checkpoint(u)
    if tensor_hashes(policy,True)!=frozen_before:raise RuntimeError('Frozen public-S0 backbone changed')
    save_json(out/'FIRST_FORWARD_DTYPES.json',first_forward)
    np.savez_compressed(out/'FIRST_FORWARD_OUTPUTS.npz',**first_outputs)
    np.save(out/'SAMPLED_ANCHORS.npy',np.asarray(samples,dtype=np.int32))
    np.save(out/'SAMPLING_BRANCHES.npy',np.asarray(sampling_branches,dtype=np.uint8))
    del opt,trainable,policy;torch.cuda.empty_cache()
    # All model reconstruction/readback occurs only after the continuous2000 ends.
    from safetensors.torch import load_model
    verified=[]
    for u in CHECKPOINTS:
        checkpoint=out/f'checkpoint_{u:04d}';final=checkpoint/'final'
        restored=SmolVLAPolicy(pcfg).to('cuda');configure_precision(restored,'B')
        load_model(restored,str(final/'model.safetensors'),strict=True,device='cuda')
        actual_hashes={**tensor_hashes(restored,False),**tensor_hashes(restored,True)}
        expected_hashes=read(checkpoint/'CHECKPOINT_PARAMETERS.json')
        if actual_hashes!=expected_hashes:raise RuntimeError('FP32 checkpoint tensor readback mismatch')
        changed=[n for n in trainable_before if actual_hashes[n]!=trainable_before[n]]
        result=dict(status='S23_WEIGHTS_ONLY_CHECKPOINT_VERIFIED',checkpoint_update=u,planned_updates=1000,planned_lr_horizon=2000,
            fresh_from_public_S0=False,prior_project_SFT_updates=2000,cumulative_project_SFT_updates=2000+u,
            single_continuous_optimizer=True,optimizer_state_reused=False,training_randomness=armcfg,
            source_checkpoint_weight_sha256=cfg['source_checkpoint']['weight_sha256'],
            precision_source='S22/2000 raw FP32 exact load; original S0 mixed map retained for inference',sampling='uniform16 over all1028 anchors',
            effective_batch=16,microbatch=a.microbatch,accumulation=16//a.microbatch,
            frozen_backbone_unchanged=True,normalizer_stats_overwritten=False,reload_weight_tensors_byte_equal=True,
            reload_forward_performed=False,changed_trainable_tensor_count=len(changed),zero_parameter_change_is_valid=True,
            original_processor_files={n:sha(final/n) for n in PROCESSORS},
            final_files={f.name:sha(f) for f in final.iterdir() if f.is_file()},
            dataset_manifest_sha256=sha(a.data/'MANIFEST.json'),training_config_sha256=sha(ROOT/'configs/s23_deadline_v1.json'),
            source_anchor_counts=read(checkpoint/'CHECKPOINT_EXPOSURE.json')['source_anchor_counts'],anchor_count_at_checkpoint=u*16,new_API_calls=0)
        save_json(checkpoint/'TRAIN_RESULT.json',result);verified.append(u)
        del restored;torch.cuda.empty_cache()
    save_json(out/'TRAIN_RESULT.json',dict(status='S23_WEIGHTS_ONLY_ADDITIONAL1000_COMPLETE',updates=1000,verified_checkpoints=verified,
        planned_checkpoints=list(CHECKPOINTS),optimizer_instances_in_main=1,optimizer_resets_at_checkpoints=0,
        fresh_from_public_S0=False,source_checkpoint_weight_sha256=cfg['source_checkpoint']['weight_sha256'],
        sampled_anchors_sha256=sha(out/'SAMPLED_ANCHORS.npy'),sampling_branches_sha256=sha(out/'SAMPLING_BRANCHES.npy'),
        frozen_backbone_unchanged=True,original_processor_files={n:sha(a.policy_snapshot/n) for n in PROCESSORS},
        normalizer_stats_overwritten=False,elapsed_s=time.monotonic()-start,new_API_calls=0))
    return 0

if __name__=='__main__':raise SystemExit(main())
