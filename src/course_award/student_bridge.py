"""Official processor boundary, adapted from PARC MeasurementStack.action (Apache-2.0).
Source path/SHA are recorded in docs/S2_STUDENT_HANDOFF_PROVENANCE.json.
No environment construction/reset/step is performed by this bridge.
"""
import numpy as np

def batch(value):
    if isinstance(value,dict):return {k:batch(v) for k,v in value.items()}
    return np.asarray(value)[None,...]

def reset_for_handoff(policy,processors,seed):
    import random,torch
    for item in (policy,*processors):item.reset()
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if any(len(q) for q in policy._queues.values()):raise RuntimeError('Policy queue not empty after reset')
