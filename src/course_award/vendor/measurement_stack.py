"""The new measurement path, using the same official factories/processors as G1."""
from contextlib import nullcontext
import torch
import gymnasium as gym
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import make_env,preprocess_observation,make_env_pre_post_processors,check_env_attributes_and_types
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy,make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed


class MeasurementStack:
    def __init__(self,cfg):
        self.cfg=cfg
        self.device=get_safe_torch_device(cfg.policy.device,log=True)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        set_seed(cfg.seed)  # same position as official eval_main, no estimated RNG skips
        self.envs=make_env(cfg.env,n_envs=cfg.eval.batch_size,use_async_envs=cfg.eval.use_async_envs,
                           trust_remote_code=cfg.trust_remote_code)
        self.policy=make_policy(cfg=cfg.policy,env_cfg=cfg.env,rename_map=cfg.rename_map)
        self.policy.eval()
        self.pre,self.post=make_pre_post_processors(policy_cfg=cfg.policy,pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={'device_processor':{'device':str(self.policy.config.device)},
                                   'rename_observations_processor':{'rename_map':cfg.rename_map}})
        self.env_pre,self.env_post=make_env_pre_post_processors(env_cfg=cfg.env,policy_cfg=cfg.policy)

    @property
    def processors(self):return [self.env_pre,self.pre,self.post,self.env_post]

    def action(self,observation,env):
        observation=preprocess_observation(observation)
        try:observation['task']=list(env.call('task_description'))
        except (AttributeError,NotImplementedError):observation['task']=list(env.call('task'))
        observation=self.env_pre(observation);observation=self.pre(observation)
        with torch.inference_mode():action=self.policy.select_action(observation)
        action=self.post(action);action=self.env_post({ACTION:action})[ACTION]
        result=action.to('cpu').numpy()
        assert result.ndim==2 and result.shape==(1,7)
        return result

    def close(self):
        for group in self.envs.values():
            for env in group.values():env.close()


def render_like_official(env):
    # Official G1 enables its render callback, even though frames are not policy inputs.
    if isinstance(env,gym.vector.SyncVectorEnv):env.envs[0].render()
    else:env.call('render')


@parser.wrap()
def new_main(cfg:EvalPipelineConfig):
    stack=MeasurementStack(cfg)
    try:
        env=next(iter(next(iter(stack.envs.values())).values()))
        stack.policy.reset()
        observation,_=env.reset(seed=range(cfg.seed,cfg.seed+1),options={NEW_ROLLOUT_OPTION:True})
        render_like_official(env)
        limit=env.call('_max_episode_steps')[0]
        check_env_attributes_and_types(env)
        with torch.no_grad(),torch.autocast(device_type=stack.device.type) if cfg.policy.use_amp else nullcontext():
            for _ in range(limit):
                action=stack.action(observation,env)
                observation,_,terminated,truncated,_=env.step(action)
                render_like_official(env)
                if bool((terminated|truncated).all()):break
    finally:stack.close()
