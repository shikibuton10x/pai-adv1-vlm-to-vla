"""S8 only: extract first settled G1 geometry, no previous policy actions/labels."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assets = json.loads((ROOT/'configs/local_assets.json').read_text())
    identity = json.loads((ROOT/'results/s8_task_identification_v1/TASK_IDENTITY.json').read_text())
    if identity['task_name'] != 'put_the_wine_bottle_on_top_of_the_cabinet' or identity['task_index'] != 2:
        raise ValueError('Public runtime task identity required')
    source = Path(assets['g1_root'])/'libero_goal_a10/trace'
    dest = ROOT/'fixtures/s8_g1/libero_goal_a10/trace'
    dest.mkdir(parents=True, exist_ok=False)
    entries = {}
    for i in (0, 1, 2):
        trace = source/f'task_02_episode_{i:03d}.npz'
        meta = trace.with_name(trace.stem+'_metadata.json')
        original = identity['refs'][str(i)]
        if sha(trace) != original['trace_sha256'] or sha(meta) != original['metadata_sha256']:
            raise RuntimeError('Original G1 source changed')
        with np.load(trace, allow_pickle=False) as z:
            position = z['state_object_pos'][:1].copy()
        out, out_meta = dest/trace.name, dest/meta.name
        np.savez_compressed(out, state_object_pos=position)
        out_meta.write_text(json.dumps(dict(object_names=original['object_names'],
            scope='first settled position row only; no rollout actions, successes or images'), indent=2)+'\n')
        entries[str(i)] = dict(source_trace=str(trace), source_trace_sha256=sha(trace),
            source_metadata=str(meta), source_metadata_sha256=sha(meta),
            trace=str(out.relative_to(ROOT)), trace_sha256=sha(out),
            metadata=str(out_meta.relative_to(ROOT)), metadata_sha256=sha(out_meta),
            extracted_field='state_object_pos[0:1]', row_sha256=original['row_sha256'],
            selected_public_init=i, base_seed=42, selected_env_seed=42+i,
            prior_reset_preparation_steps=10*i, selected_settle_steps=10)
    provenance = dict(version='S8_CABINET_MINIMAL_G1_REFERENCE_V1', suite='libero_goal',
        task_index=2, task_name=identity['task_name'], entries=entries,
        public_init_path=identity['init_path'], public_init_sha256=identity['init_sha256'],
        public_init_shape=identity['init_shape'], public_init_dtype=identity['init_dtype'],
        public_bddl_path=identity['bddl_path'], public_bddl_sha256=identity['bddl_sha256'],
        license='G1 experiment reference: no explicit license located; authorized course-internal minimal reuse. Public LIBERO code/assets license is recorded separately; G1 is not relabeled MIT.',
        source_read_only=True, actions_copied=False, old_labels_used=False,
        physics_steps=0, new_API_calls=0, new_GPU_pods=0)
    (ROOT/'docs/S8_G1_REFERENCE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    assets['g1_root'] = str(ROOT/'fixtures/s8_g1')
    (ROOT/'configs/local_assets_s8.json').write_text(json.dumps(assets, indent=2)+'\n')
    print(json.dumps(dict(task_index=2, requested_inits=[0,1,2],
        copied_files=6, copied_bytes=sum(p.stat().st_size for p in dest.iterdir()), physics_steps=0)))


if __name__ == '__main__':
    main()
