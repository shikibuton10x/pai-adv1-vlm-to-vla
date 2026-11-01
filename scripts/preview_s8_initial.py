"""Authorized S8 initial-state preview only: build/settle, no further action."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from course_award.env import build, sha256
from course_award.cabinet_s8 import CabinetEpisode, initial_identity, native_geometry
from course_award.side_robot_card import capture
from run_s0 import write_json
from run_s1 import source_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', type=int, choices=[0,1,2], required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError('Dedicated S8 output required')
    out.mkdir(parents=True, exist_ok=False)
    asset = ROOT/'configs/local_assets_s8.json'
    source_snapshot(out, asset, extra_files=(Path(__file__),
        ROOT/'docs/S8_CABINET_COMPARISON_CONTRACT.md',
        ROOT/'docs/S8_G1_REFERENCE_PROVENANCE.json'))
    env = None
    try:
        env, raw, receipt = build(json.loads(asset.read_text()), 'libero_goal', 2, a.init, out, render=True)
        identity = initial_identity(receipt, a.init)
        write_json(out/'INITIAL_IDENTITY.json', identity)
        ep = CabinetEpisode(env)
        if not ep.records[0]['nontarget']['nontarget_preserved']:
            raise RuntimeError('Initial S8 preservation failure')
        paths, views, card, proof = capture(env, out/'initial_views', 0)
        write_json(out/'INITIAL_VIEWS.json', dict(paths=list(map(str,paths)),
            views=views, robot_card=card, proof=proof, additional_actions=0))
        write_json(out/'NATIVE_GEOMETRY.json', native_geometry(env))
        np.savez_compressed(out/'RAW_INITIAL_OBSERVATION.npz',
            raw_eef_pos=raw['robot0_eef_pos'], raw_eef_quat=raw['robot0_eef_quat'],
            raw_gripper_qpos=raw['robot0_gripper_qpos'],
            raw_agent=raw['agentview_image'], raw_wrist=raw['robot0_eye_in_hand_image'])
        if not ep.save(out):
            raise RuntimeError('Preview changed physics')
        result = dict(status='INITIAL_ONLY_PREVIEW_PASS', init=a.init,
            selected_settle_steps=10, prior_preparation_steps=10*a.init,
            all_process_explicit_steps=10*(a.init+1), post_settle_actions=0,
            native=ep.records[0]['native_success'],
            primary=ep.records[0]['nontarget']['nontarget_preserved'],
            drawer=ep.records[0]['drawer_diagnostic'], new_API_calls=0, GPU=0,
            source_sha256=sha256(Path(__file__)))
        write_json(out/'RESULT.json', result)
        print(json.dumps(result))
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
