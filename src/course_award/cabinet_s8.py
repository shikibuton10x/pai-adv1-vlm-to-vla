"""S8 cabinet task identity, episode and one predetermined terminal observation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .env import Monitor
from .cabinet_quality_s8 import CabinetMonitor, TARGET, RECEIVER, REGION
from .lift import LiftMonitor
from .contact_diagnostic import self_contact_flags
from .nontarget_v2 import HandEdge
from .recovery_episode_s6 import Episode as ExistingEpisode
from .full_task_s7 import ReleaseAudit
from .goal_completion_v4 import GoalExecutor
from .control import pose, limited, normalized_action
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'S8_CABINET_COMPARISON_V1'
TASK = 'put_the_wine_bottle_on_top_of_the_cabinet'
POLICY_STEPS, OBSERVATION_STEPS = 950, 40


def candidate(row, gripper):
    return bool(np.isfinite(gripper) and gripper < 0. and
                row['nontarget']['nontarget_preserved'] and row['native_success']
                and not row['target']['contacts'])


class CabinetEpisode(ExistingEpisode):
    """Existing observation/save methods; swap only exact receiver monitor."""
    def __init__(self, env):
        self.env = env
        self.legacy, self.nt, self.lift = Monitor(env), CabinetMonitor(env), LiftMonitor(env)
        self.actions, self.states, self.records = [], [], []
        self.valid = True
        self.edge = HandEdge(self_contact_flags(env.env)['hand'])
        self.drawer_names = [RECEIVER+'_'+part+'_level' for part in ('top', 'middle', 'bottom')]
        self.drawer_addresses = [int(env.env.sim.model.get_joint_qpos_addr(n)) for n in self.drawer_names]
        self.drawer_base = env.env.sim.data.qpos[self.drawer_addresses].copy()
        self.drawer_peak = np.zeros(3)
        self.observe(None, 'initial')

    def observe(self, action, phase):
        row = super().observe(action, phase)
        values = self.env.env.sim.data.qpos[self.drawer_addresses].copy()
        self.drawer_peak = np.maximum(self.drawer_peak, np.abs(values-self.drawer_base))
        row['drawer_diagnostic'] = dict(joint_names=self.drawer_names,
            qpos_m=values.tolist(), initial_qpos_m=self.drawer_base.tolist(),
            max_abs_shift_m=self.drawer_peak.tolist(),
            within_3mm=bool(np.all(self.drawer_peak <= .003+1e-12)),
            primary_stop=False, teacher_input=False)
        return row


class CabinetExecutor(GoalExecutor):
    def __init__(self, env, after_step):
        super().__init__(env, after_step)
        self.action_limit = POLICY_STEPS
        self.controller.update(version=VERSION, policy_action_cap=POLICY_STEPS,
                               common_terminal_observation_steps=OBSERVATION_STEPS)


def initial_identity(receipt, initial_id):
    provenance = json.loads((ROOT/'docs/S8_G1_REFERENCE_PROVENANCE.json').read_text())
    entry = provenance['entries'][str(initial_id)]
    if (receipt['task_id'], receipt['task_name'], receipt['initial_state_id']) != (2, TASK, initial_id):
        raise RuntimeError('Runtime suite task/init differs from S8')
    if receipt['bddl_sha256'] != provenance['public_bddl_sha256'] or receipt['init_sha256'] != provenance['public_init_sha256']:
        raise RuntimeError('S8 public task/init file changed')
    if (receipt['g1_reference_sha256'] != entry['trace_sha256'] or
            receipt['seed'] != 42+initial_id or
            [r['seed'] for r in receipt['reset_history']] != list(range(42,43+initial_id))):
        raise RuntimeError('G1 identity/reset sequence changed')
    return dict(receipt, public_row_sha256=entry['row_sha256'], base_env_seed=42,
        selected_env_seed=42+initial_id, policy_seed=42, selected_settle_steps=10,
        prior_preparation_control_steps=10*initial_id,
        bddl_language='Put the wine bottle on the top of the drawer',
        task_language='put the wine bottle on top of the cabinet',
        intended_goal=['On', TARGET, REGION], receiver=RECEIVER)


def native_geometry(env):
    d = env.env
    site = d.sim.model.site_name2id(REGION)
    return dict(diagnostic_only=True, teacher_input=False, target_body_root=TARGET,
        receiver_root=RECEIVER, region=REGION, site_id=int(site),
        site_body_id=int(d.sim.model.site_bodyid[site]),
        site_size=d.sim.model.site_size[site].tolist(), site_pos=d.sim.model.site_pos[site].tolist(),
        site_quat=d.sim.model.site_quat[site].tolist(),
        initial_site_xpos=d.sim.data.site_xpos[site].tolist(),
        initial_site_xmat=d.sim.data.site_xmat[site].tolist(),
        native_semantics='Unchanged LIBERO On/check_ontop; separate release-stability criterion.')


class FixedObservation:
    def __init__(self, *, candidate_found, start_step, gripper):
        self.candidate_found = bool(candidate_found)
        self.start_step = int(start_step)
        self.gripper = -1. if candidate_found else float(gripper)
        if not np.isfinite(self.gripper):
            raise ValueError('Last actual finite gripper scalar required')
        self.audit = None
        self.steps = 0

    def add(self, step, sim_time, row, position, quaternion, *, dt):
        if step != self.start_step+self.steps+1 or self.steps >= 40:
            raise ValueError('Exactly the next predetermined40 states required')
        self.steps += 1
        # Reuse the audit's binary OPEN code for scoring only. The actual
        # applied noncandidate raw scalar remains unchanged, including zero.
        audit_gripper = -1. if self.gripper < 0 else self.gripper
        if self.steps == 20:
            self.audit = ReleaseAudit(step, sim_time, row, position, quaternion,
                                      audit_gripper, dt=dt)
        elif self.steps > 20:
            self.audit.add(step, sim_time, row, position, quaternion, audit_gripper)

    def result(self):
        audit = self.audit.result() if self.audit else dict(full_task_success=False, states=[])
        measured = audit.pop('full_task_success')
        return dict(candidate_found=self.candidate_found, observation_steps=self.steps,
            observation_start_step=self.start_step, measurement_anchor_step=self.start_step+20,
            expected_last_step=self.start_step+40, gripper_command=self.gripper,
            open_command_definition='latest actual raw scalar<0; audit binary code only, no physical scalar replacement',
            release_stability_pass=bool(self.candidate_found and self.steps == 40 and measured),
            noncandidate_cannot_be_promoted=True, audit=audit)


def run_observation(env, ep, recorder, raw, *, gripper, candidate_found, deadline):
    """One fixed pose throughout40; no API, policy, queue consumption or replan."""
    start = len(ep.actions)
    if start > 950 or start+40 > 990:
        raise ValueError('1000 incl settle:950 policy +40 observation only')
    p0, r0 = pose(env.env)
    p0.setflags(write=False)
    r0.setflags(write=False)
    observer = FixedObservation(candidate_found=candidate_found, start_step=start, gripper=gripper)
    target = ep.nt.names.index(TARGET)
    reason = 'COMMON_OBSERVATION_COMPLETE'
    for i in range(40):
        if time.monotonic() >= deadline-30:
            reason = 'WALL_TIME_LIMIT'; break
        p, r = pose(env.env)
        # Keep the six-axis normalization exactly as before. The policy's raw
        # seventh scalar can be continuous/zero/outside[-1,1]; preserve it for
        # the SAME existing env/gripper formatting, without opening/clipping it.
        action = normalized_action(limited(p0-p, .004),
            limited(Rotation.from_matrix(r0@r.T).as_rotvec(), .03), -1.)
        action[6] = observer.gripper
        raw, *_ = env.step(action)
        row = ep.observe(action, 'common_observer_settling' if i < 20 else 'common_observer_measurement')
        observer.add(len(ep.actions), float(env.env.sim.data.time), row,
            ep.states[-1]['object_pos'][target], ep.states[-1]['object_quat'][target],
            dt=float(env.env.sim.model.opt.timestep))
        recorder.record(env, raw, row, gripper=observer.gripper, decision_id=None)
        if not row['nontarget']['nontarget_preserved']:
            reason = 'NONTARGET_VIOLATION'; break
    return dict(observer.result(), status=reason,
        fixed_eef_position_world_m=p0.tolist(), fixed_eef_rotation_world=r0.tolist(),
        policy_actions_at_start=start, new_API_calls=0, new_policy_calls=0,
        no_reanchor=True, latest_policy_gripper=float(gripper)), raw


def score(ep, observation, state_valid):
    last = ep.records[-1]
    preserved = bool(last['nontarget']['nontarget_preserved'])
    stable = bool(state_valid and observation and observation['release_stability_pass'] and preserved)
    drawers = last['drawer_diagnostic']['within_3mm']
    return dict(native_success=bool(last['native_success']),
        first_native_step=next((r['step'] for r in ep.records if r['native_success']), None),
        nontarget_preserved=preserved,
        strict_nontarget_preserved=bool(last['nontarget']['strict_contact_diagnostic']['nontarget_preserved']),
        legacy_clean=bool(last['legacy']['clean_so_far']), release_stability_pass=stable,
        full_task_success=stable, drawer_diagnostic=last['drawer_diagnostic'],
        teaching_material_candidate=bool(stable and drawers),
        moving_lift=bool(last['lift']['lift_confirmed']), lift_evidence=last['lift'])


def load_spec():
    from .cabinet_quality_s8 import SETTINGS
    cfg = json.loads((ROOT/'configs/s8_cabinet_comparison_v1.json').read_text())
    if (cfg['version'], cfg['task_index'], cfg['task_name'], cfg['target'], cfg['receiver'], cfg['region'],
        cfg['settle_steps'], cfg['policy_steps'], cfg['terminal_observation_steps'], cfg['total_physics_cap']) != (
        VERSION, 2, TASK, TARGET, RECEIVER, REGION, 10, 950, 40, 1000):
        raise ValueError('S8 public task/physical contract changed')
    t = cfg['teacher']
    if (t['model'],t['thinking_level'],t['input_cap'],t['output_cap'],t['http_timeout_s'],
        t['outer_timeout_s'],t['cell_generations'],t['cell_money_cap_micro'],
        t['stage_generations'],t['stage_money_cap_micro'],t['common_generation_cap'],
        t['common_money_cap_micro']) != ('gemini-4-argon','MEDIUM',9000,32768,900,2700,24,1500000,72,3000000,193,5905310):
        raise ValueError('S8 prospectively fixed inference/budget contract changed')
    if cfg['evaluation_profile'] != SETTINGS or cfg['n_action_steps'] != 50:
        raise ValueError('S8 quality/student boundary changed')
    return cfg


def snapshot_source(out, extra=()):
    """Small dependency snapshot; no old results, credentials, models or caches."""
    from .env import sha256
    files = [*sorted((ROOT/'src').rglob('*.py')),
        *[ROOT/p for p in ('scripts/run_s0.py', 'scripts/run_s8_teacher.py',
            'scripts/launch_s8_teacher.py', 'configs/local_assets_s8.json',
            'configs/s8_cabinet_comparison_v1.json',
            'docs/S8_CABINET_COMPARISON_CONTRACT.md', 'docs/S8_G1_REFERENCE_PROVENANCE.json')],
        *sorted((ROOT/'fixtures/s8_g1').rglob('*.*')), *extra]
    manifest = {}
    for path in sorted(set(files)):
        dest = Path(out)/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest[str(path.relative_to(ROOT))] = sha256(dest)
    (Path(out)/'source_snapshot/MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
