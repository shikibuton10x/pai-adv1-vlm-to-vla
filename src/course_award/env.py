"""Explicit-path public LIBERO adapter, adapted from the recorded teleop adapter.

See docs/SOURCE_PROVENANCE.json. Public assets are read-only; all new state lives
under the current experiment root. The quality baseline starts after settling.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .safety import Protection, physical_receiver


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(config, out, *, render):
    root = Path(config['project_root']).resolve()
    out = Path(out).resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('Output must be a child of the independent project root')
    libero = Path(config['libero_root']).resolve()
    source = libero / 'libero/libero'
    if not all((source / p).is_dir() for p in ['assets', 'init_files', 'bddl_files']):
        raise FileNotFoundError('Explicit public LIBERO assets are incomplete')
    context = out / 'context'
    context.mkdir(parents=True, exist_ok=False)
    for name in ['matplotlib', 'numba', 'torch', 'xdg']:
        (out / 'cache' / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        LIBERO_CONFIG_PATH=str(context), MUJOCO_GL='glfw' if render else 'disable',
        CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', WANDB_MODE='disabled',
        MPLCONFIGDIR=str(out / 'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(out / 'cache/numba'),
        TORCH_HOME=str(out / 'cache/torch'), XDG_CACHE_HOME=str(out / 'cache/xdg'),
    )
    import yaml
    (context / 'config.yaml').write_text(yaml.safe_dump({
        'benchmark_root': str(source), 'bddl_files': str(source / 'bddl_files'),
        'init_states': str(source / 'init_files'), 'assets': str(source / 'assets'),
        'datasets': str(out / 'datasets_unused'),
    }))
    sys.path.insert(0, str(libero))


def build(config, suite_name, task_id, initial_id, out, *, render=True):
    configure(config, out, render=render)
    if suite_name not in ('libero_goal', 'libero_10'):
        raise ValueError('Unsupported public suite')
    if type(task_id) is not int or type(initial_id) is not int:
        raise ValueError('Task/initial indices must be integers')
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError('Task index outside suite')
    task = suite.get_task(task_id)
    init_path = Path(get_libero_path('init_states')) / task.problem_folder / Path(task.init_states_file).name
    init_sha = sha256(init_path)
    allowed = [(np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               np.ndarray, np.dtype, type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        states = np.asarray(torch.load(init_path, map_location='cpu', weights_only=True))
    if sha256(init_path) != init_sha or states.dtype != np.float64 or states.ndim != 2:
        raise RuntimeError('Public initial-state identity/type changed')
    if not 0 <= initial_id < len(states):
        raise ValueError('Exact initial-state index required; no modulo')
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    reference = (Path(config['g1_root']) / (suite_name + '_a10') / 'trace' /
                 f'task_{task_id:02d}_episode_{initial_id:03d}.npz')
    if not reference.is_file():
        raise FileNotFoundError('Explicit G1 reference unavailable')
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=render,
                     has_renderer=False, has_offscreen_renderer=render,
                     camera_heights=360, camera_widths=360, hard_reset=True,
                     ignore_done=True, control_freq=20, horizon=300)
    try:
        # Same successful reset lifecycle as the original wrapper, but random
        # sampling failures propagate instead of retrying without a bound.
        env.env.reset()
        reset_history = []
        for episode in range(initial_id + 1):
            env.seed(42 + episode)
            env.env.reset()
            obs = env.set_init_state(states[episode].copy())
            for _ in range(10):
                obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            reset_history.append(dict(episode=episode, seed=42 + episode,
                initial_state_id=episode,
                property_initializers=len(env.env.object_property_initializers)))
        source_meta = json.loads(reference.with_name(reference.stem + '_metadata.json').read_text())
        names = source_meta['object_names']
        ids = [env.env.obj_body_id[n] for n in names]
        with np.load(reference, allow_pickle=False) as z:
            error = float(np.abs(env.env.sim.data.body_xpos[ids] - z['state_object_pos'][0]).max())
        if error > 1e-6:
            raise RuntimeError(f'G1_INITIAL_GEOMETRY_IDENTITY_FAILED max_abs_m={error}')
        receipt = dict(suite=suite_name, task_id=task_id, task_name=task.name,
            initial_state_id=initial_id, instruction=task.language, seed=42 + initial_id,
            bddl_path=str(bddl), bddl_sha256=sha256(bddl), init_sha256=init_sha,
            g1_reference_path=str(reference), g1_reference_sha256=sha256(reference),
            initial_object_position_max_abs_m=error, initial_identity_tolerance_m=1e-6,
            initial_identity_pass=True, reset_history=reset_history,
            reset_retry_policy='no adapter retries', control_hz=20,
            preparation_settle_steps=10 * (initial_id + 1),
            quality_start='step0 after 10 settling steps of the requested initial state',
            preparation_quality_measured=False, rendered=render, weights_only=True,
            asset_roots_read_only=True)
        return env, obs, receipt
    except BaseException:
        env.close()
        raise


def metadata(d):
    names = list(d.obj_body_id)
    return dict(object_names=names, object_body_ids=[int(d.obj_body_id[n]) for n in names],
        movable_object_names=list(d.objects_dict), fixture_names=list(d.fixtures_dict),
        geom_names=list(d.sim.model.geom_names),
        geom_body_ids=list(map(int, d.sim.model.geom_bodyid)),
        body_parent_ids=list(map(int, d.sim.model.body_parentid)),
        parsed_problem=d.parsed_problem)


def contact_pairs(d, meta, protected):
    names = meta['geom_names']
    robot = {i for i, n in enumerate(names) if n.startswith(('robot0_', 'gripper0_'))}
    rows = []
    for name in protected:
        root = meta['object_body_ids'][meta['object_names'].index(name)]
        descendants = {root}
        for i, parent in enumerate(meta['body_parent_ids']):
            if parent in descendants:
                descendants.add(i)
        geoms = {i for i, body in enumerate(meta['geom_body_ids']) if body in descendants}
        for contact in d.sim.data.contact[:d.sim.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if b in robot:
                a, b = b, a
            if a in robot and b in geoms and float(contact.dist) <= 0:
                rows.append(dict(object=name, robot_geom=names[a], object_geom=names[b],
                                 distance_m=float(contact.dist)))
    return rows


class Monitor:
    """Cumulative quality from settled episode start; segment forks keep baseline.

    A fork clears only its own event history. It cannot clear the parent history.
    Both histories use the same original/goal-completion reference positions.
    """
    def __init__(self, env):
        self.env = env
        self.d = env.env
        self.meta = metadata(self.d)
        self.names = self.meta['object_names']
        self.ids = [self.d.obj_body_id[n] for n in self.names]
        self.reference_positions = self.d.sim.data.body_xpos[self.ids].copy()
        self.reference_quaternions = self.d.sim.data.body_xquat[self.ids].copy()
        self.guard = Protection(self.names, self.reference_positions, self.reference_quaternions)
        self.step = self.start_step = 0
        self.first_native = self.first_contact = None
        self.first_contacts = {}
        self.manipulating, self.completed = set(), set()

    def fork_segment(self):
        other = Monitor(self.env)
        other.reference_positions = self.reference_positions.copy()
        other.reference_quaternions = self.reference_quaternions.copy()
        other.guard = Protection(self.names, other.reference_positions, other.reference_quaternions)
        for field in ['base', 'base_up', 'epoch']:
            setattr(other.guard, field, getattr(self.guard, field).copy())
        other.guard.completed = self.guard.completed.copy()
        other.manipulating, other.completed = self.manipulating.copy(), self.completed.copy()
        other.step = other.start_step = self.step
        return other

    def observe(self, action=None):
        d = self.d
        goals = d.parsed_problem['goal_state']
        goal = (list(g for g in goals if not d._eval_predicate(g)) or goals)[0]
        active = goal[1]
        receiver = physical_receiver(goal, self.names)
        if receiver is None and len(goal) > 2 and goal[0].lower() in ('on', 'in'):
            resolved = d.parsed_problem['regions'].get(goal[2], {}).get('target')
            if resolved in self.names:
                receiver = resolved
        if action is not None:
            self.step += 1
            if action[6] > 0:
                self.manipulating.add(active)
        state = self.guard.observe(d.sim.data.body_xpos[self.ids],
            d.sim.data.body_xquat[self.ids], active=active,
            stage='manipulation' if active in self.manipulating else 'approach',
            receiver=receiver, step=self.step)
        protected = [n for n, role in zip(self.names, state['protection_role']) if role != 0]
        points = contact_pairs(d, self.meta, protected)
        if points and self.first_contact is None:
            self.first_contact = dict(step=self.step, pairs=points)
        for point in points:
            self.first_contacts.setdefault(point['object'], dict(step=self.step, pair=point))
        success = bool(self.env.check_success())
        if success and self.first_native is None:
            self.first_native = self.step
        for g in goals:
            if g[1] in self.names and d._eval_predicate(g) and g[1] not in self.completed:
                self.guard.complete(g[1], d.sim.data.body_xpos[self.ids], d.sim.data.body_xquat[self.ids])
                self.completed.add(g[1])
        return dict(step=self.step, segment_start_step=self.start_step, active=active,
            receiver=receiver, native_success=success, first_native_step=self.first_native,
            protection_ok=bool(state['protection_ok']),
            first_protection_violation=copy.deepcopy(self.guard.violation),
            current_translation_mm=state['protection_translation_mm'].tolist(),
            current_tilt_deg=state['protection_tilt_deg'].tolist(),
            object_names=self.names, movable_object_names=self.meta['movable_object_names'],
            fixture_names=self.meta['fixture_names'], roles=state['protection_role'].tolist(),
            contacts=points, first_contact=copy.deepcopy(self.first_contact),
            first_contact_by_object=copy.deepcopy(self.first_contacts),
            clean_so_far=bool(state['protection_ok'] and self.first_contact is None),
            active_manipulation_definition='first positive close command, NOT proof of grasp',
            reference='settled episode start; completed-object epochs carried across segment forks',
            contact_sampling='signed distance<=0 at 20Hz control states; no substep audit',
            fixture_contact_scope='named fixtures included; table/workspace excluded')
