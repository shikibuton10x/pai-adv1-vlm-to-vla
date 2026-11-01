from dataclasses import dataclass
import numpy as np
STRICT=1
SOFT=2
EXEMPT=0
def up_axis_wxyz(q):
    q=np.asarray(q,dtype=float)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=-1)
def physical_receiver(goal, movable_names):
    return goal[2] if goal[0].lower() in ('on', 'in') and goal[2] in movable_names else None

@dataclass(frozen=True)
class Limits:
    strict_mm: float = 1.0
    soft_mm: float = 3.0
    tilt_deg: float = 3.0
    numerical_roundoff: float = 1e-09

class Protection:

    def __init__(self, names, positions, quaternions):
        self.names = list(names)
        self.limits = Limits()
        self.original = np.asarray(positions).copy()
        self.base = self.original.copy()
        self.original_up = up_axis_wxyz(quaternions)
        self.base_up = self.original_up.copy()
        self.epoch = np.zeros(len(names), dtype=np.int64)
        self.peak_mm = np.zeros((2, len(names)))
        self.peak_deg = np.zeros((2, len(names)))
        self.violation = None
        self.completed = set()

    def complete(self, name, positions, quaternions):
        index = self.names.index(name)
        self.base[index] = np.asarray(positions)[index]
        self.base_up[index] = up_axis_wxyz(quaternions)[index]
        self.epoch[index] += 1
        self.completed.add(name)

    def observe(self, positions, quaternions, *, active, stage, receiver, step):
        positions = np.asarray(positions)
        up = up_axis_wxyz(quaternions)
        shift = np.linalg.norm(positions - self.base, axis=-1) * 1000
        tilt = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(self.base_up, up), axis=-1), np.clip(np.sum(self.base_up * up, axis=-1), -1, 1)))
        roles = np.full(len(self.names), STRICT, dtype=np.int8)
        if receiver in self.names:
            roles[self.names.index(receiver)] = SOFT
        if active in self.names and active not in self.completed:
            roles[self.names.index(active)] = SOFT if stage == 'approach' else EXEMPT
        for index, name in enumerate(self.names):
            role = int(roles[index])
            if role == EXEMPT:
                continue
            j = role - 1
            self.peak_mm[j, index] = max(self.peak_mm[j, index], shift[index])
            self.peak_deg[j, index] = max(self.peak_deg[j, index], tilt[index])
            limit = self.limits.strict_mm if role == STRICT else self.limits.soft_mm
            exceeded = shift[index] > limit + self.limits.numerical_roundoff or tilt[index] > self.limits.tilt_deg + self.limits.numerical_roundoff
            if exceeded and self.violation is None:
                self.violation = dict(step=int(step), object=name, role='strict' if role == STRICT else 'soft', translation_mm=float(shift[index]), tilt_deg=float(tilt[index]), translation_limit_mm=limit, tilt_limit_deg=self.limits.tilt_deg, reference_epoch=int(self.epoch[index]), active=active, stage=stage)
        return dict(protection_role=roles, protection_reference_epoch=self.epoch.copy(), protection_translation_mm=shift, protection_tilt_deg=tilt, strict_translation_peak_mm=self.peak_mm[0].copy(), soft_translation_peak_mm=self.peak_mm[1].copy(), strict_tilt_peak_deg=self.peak_deg[0].copy(), soft_tilt_peak_deg=self.peak_deg[1].copy(), rewind_translation_mm=np.linalg.norm(positions - self.original, axis=-1) * 1000, rewind_tilt_deg=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(self.original_up, up), axis=-1), np.clip(np.sum(self.original_up * up, axis=-1), -1, 1))), protection_ok=self.violation is None)
