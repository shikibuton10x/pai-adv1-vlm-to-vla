"""Run one local, no-provider Goal9/init0 connection check and preserve evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from PIL import Image, ImageDraw

from course_award.env import build, Monitor, sha256
from course_award.control import BoundedExecutor, Motion, pose


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def robot_state(env):
    d = env.env
    p, r = pose(d)
    robot = d.robots[0]
    return dict(eef_world_m=p.tolist(), eef_rotation_world=r.tolist(),
        gripper_qpos=d.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].tolist(),
        gripper_current_action=robot.gripper.current_action.tolist(),
        sim_time_s=float(d.sim.data.time))


def snapshot(env, monitor):
    d = env.env
    p, r = pose(d)
    return {k: np.asarray(getattr(d.sim.data, k)).copy() for k in ['qpos', 'qvel', 'act', 'ctrl']} | dict(
        time=np.asarray(d.sim.data.time), object_pos=d.sim.data.body_xpos[monitor.ids].copy(),
        object_quat=d.sim.data.body_xquat[monitor.ids].copy(), eef_pos=p, eef_rotation=r,
        gripper_qpos=d.sim.data.qpos[d.robots[0]._ref_gripper_joint_pos_indexes].copy())


def save_views(out, label, obs, step):
    rows, natural = [], []
    for camera, key in [('agentview', 'agentview_image'), ('wrist', 'robot0_eye_in_hand_image')]:
        raw = np.asarray(obs[key]).copy()
        if raw.shape != (360, 360, 3) or raw.dtype != np.uint8:
            raise RuntimeError('Unexpected camera contract: uint8 HWC RGB 360x360 required')
        view = np.ascontiguousarray(raw[::-1])
        paths = {}
        for kind, rgb in [('raw', raw), ('natural', view)]:
            path = out / 'observations' / f'{label}_{camera}_{kind}.png'
            Image.fromarray(rgb).save(path)
            paths[kind] = dict(file=str(path.relative_to(out)), sha256=sha256(path),
                              rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        rows.append(dict(label=label, step=step, camera=camera, source_key=key,
            raw_shape=list(raw.shape), raw_dtype=str(raw.dtype), paths=paths,
            natural_transform='raw[::-1] vertical flip only; exactly once',
            student_transform_not_applied='student input uses raw[::-1,::-1]; no student in S0'))
        natural.append(Image.fromarray(view))
    return rows, natural


def timeout_handler(*unused):
    raise TimeoutError('S0 local process wall-time limit: 180 seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, default=ROOT / 'configs/local_assets.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError('New outputs must be below the independent experiment root')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'observations').mkdir()
    config = json.loads(args.assets.read_text())
    if Path(config['project_root']).resolve() != ROOT:
        raise ValueError('Configured experiment root differs from this script')
    start = time.monotonic()
    env = None
    actions, states, logs, view_rows = [], [], [], []
    result = dict(status='RUNNING', stage='S0', suite='libero_goal', task_index=9,
        initial_state_index=0, start_utc=datetime.now(timezone.utc).isoformat(),
        gpu_used=False, api_calls=0, training_run=False, student_inference=False,
        task_success_claimed=False, output=str(out))
    write_json(out / 'RUN_SPEC.json', dict(**result, intended_delta_world_m=[0, 0, .006],
        max_control_steps=10, gripper='open', quality_thresholds=dict(strict_mm=1, soft_mm=3, tilt_deg=3),
        quality_window='after 10 settling steps; settling itself is not audited',
        source_files={str(p.relative_to(ROOT)):sha256(p) for p in
            [*sorted((ROOT/'src/course_award').glob('*.py')), Path(__file__), args.assets]}))
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)
    try:
        env, obs, receipt = build(config, 'libero_goal', 9, 0, out, render=True)
        result['environment'] = receipt
        result['runtime'] = {n:importlib.metadata.version(n) for n in
            ['numpy', 'scipy', 'torch', 'mujoco', 'robosuite', 'Pillow']}
        result['python'] = sys.version
        whole = Monitor(env)
        initial_quality = whole.observe()
        segment = whole.fork_segment()
        segment_initial = segment.observe()
        result['intervention_start_step'] = whole.step
        result['prefix_clean_at_intervention'] = initial_quality['clean_so_far']
        result['initial_robot'] = robot_state(env)
        rows, initial_views = save_views(out, '000_initial', obs, 0)
        view_rows += rows
        states.append(snapshot(env, whole))
        logs.append(dict(action=None, whole=initial_quality, intervention=segment_initial,
                         robot=result['initial_robot']))

        def after_step(action, observation):
            actions.append(action.copy())
            full_status, segment_status = whole.observe(action), segment.observe(action)
            states.append(snapshot(env, whole))
            logs.append(dict(action=action.tolist(), whole=full_status,
                             intervention=segment_status, robot=robot_state(env)))
            return bool(full_status['clean_so_far'] and segment_status['clean_so_far'])

        executor = BoundedExecutor(env, after_step, total_step_limit=10)
        result['controller'] = executor.controller
        final_obs, motion = executor.execute(Motion((0., 0., .006), max_steps=10))
        if final_obs is None:
            raise RuntimeError('S0 must execute at least one action and return a new observation')
        result['motion'] = motion
        result['final_robot'] = robot_state(env)
        result['whole_history'] = logs[-1]['whole']
        result['intervention_history'] = logs[-1]['intervention']
        rows, final_views = save_views(out, '001_after_move', final_obs, executor.steps)
        view_rows += rows
        sheet = Image.new('RGB', (720, 768), (24, 28, 32))
        draw = ImageDraw.Draw(sheet)
        for row, (label, views) in enumerate([('Initial (after settle)', initial_views), ('After bounded upward move', final_views)]):
            for col, (camera, view) in enumerate(zip(['Agent view', 'Wrist view'], views)):
                draw.text((col*360+8, row*384+5), f'{label} / {camera}', fill='white')
                sheet.paste(view, (col*360, row*384+24))
        sheet.save(out / 'S0_CONTACT_SHEET.png')
        checks = dict(initial_geometry=receipt['initial_identity_pass'],
            nonzero_steps=0 < len(actions) <= 10,
            cumulative_quality=bool(result['whole_history']['clean_so_far']),
            intervention_quality=bool(result['intervention_history']['clean_so_far']),
            upward_eef_motion=motion['actual_delta_world_m'][2] > .001,
            normalized_finite_actions=bool(np.isfinite(actions).all() and np.max(np.abs(actions))<=1),
            open_only=bool(np.all(np.asarray(actions)[:,6] == -1)),
            reobserved_images=all(view_rows[i]['paths']['raw']['rgb_sha256'] != view_rows[i+2]['paths']['raw']['rgb_sha256'] for i in range(2)))
        result['checks'] = checks
        result['status'] = 'S0_PASS' if all(checks.values()) else 'S0_FAILED'
        result['observations'] = view_rows
        print(json.dumps(dict(status=result['status'], steps=len(actions),
            actual_delta_world_m=motion['actual_delta_world_m'],
            geometry_max_abs_m=receipt['initial_object_position_max_abs_m'],
            checks=checks, output=str(out))))
    except BaseException as error:
        result.update(status='S0_ERROR', error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc())
        print(result['error'], file=sys.stderr)
        raise
    finally:
        signal.alarm(0)
        if env is not None:
            env.close()
        if states:
            arrays = {k:np.stack([s[k] for s in states]) for k in states[0]}
            arrays['actions'] = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
            np.savez_compressed(out / 'trace.npz', **arrays)
            result['trace_sha256'] = sha256(out / 'trace.npz')
        with (out/'steps.jsonl').open('x') as handle:
            for row in logs:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        result['elapsed_wall_s'] = time.monotonic() - start
        result['recorded_actions'] = len(actions)
        write_json(out / 'RESULT.json', result)
    return 0 if result['status'] == 'S0_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
