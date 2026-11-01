"""S8 explicit cabinet receiver; S4/Goal9 defaults and artifacts are untouched.

Only robot contact with wooden_cabinet_1 is record-only. The same pose guard
still protects its root (3mm/3deg); drawer slide joints are diagnostic qpos only.
"""
import copy
from .nontarget_v2 import NonTargetProtection, NonTargetMonitor

PROFILE = 'S8_CABINET_RECEIVER_CONTACT_V1'
STRICT_PROFILE = 'S8_ALL_NONTARGET_CONTACT_STRICT_DIAGNOSTIC'
TARGET = 'wine_bottle_1'
RECEIVER = 'wooden_cabinet_1'
REGION = 'wooden_cabinet_1_top_side'
SETTINGS = dict(version=PROFILE, target_object=TARGET, receiver_object=RECEIVER,
    receiver_robot_contact='record_only', receiver_translation_limit_mm=3.,
    tilt_limit_deg=3., other_translation_limit_mm=1., other_robot_contact='stop',
    strict_diagnostic=STRICT_PROFILE,
    scope='named body roots; cabinet slide-joint changes, table and physics substeps are not primary pose checks')


class CabinetProtection(NonTargetProtection):
    def __init__(self, names, positions, quaternions, *, target=TARGET, receiver=RECEIVER):
        if target != TARGET or receiver != RECEIVER or receiver not in names:
            raise ValueError('S8 requires exact bottle/cabinet IDs')
        super().__init__(names, positions, quaternions, target=target, receiver=receiver)
        self.strict = NonTargetProtection(names, positions, quaternions,
                                         target=target, receiver=receiver)
        self.receiver_contact_states = 0

    def observe(self, positions, quaternions, contacts, step):
        allowed = [p for p in contacts if p['object'] == RECEIVER]
        primary = super().observe(positions, quaternions,
            [p for p in contacts if p['object'] != RECEIVER], step)
        if primary['roles'][self.names.index(RECEIVER)] != 2:
            raise RuntimeError('Cabinet must remain the soft receiver role')
        strict = self.strict.observe(positions, quaternions, contacts, step)
        if allowed:
            self.receiver_contact_states += 1
        primary.update(evaluation_profile=PROFILE,
            receiver_contact_policy='only explicit wooden_cabinet_1 robot contact record-only',
            receiver_contacts=copy.deepcopy(allowed),
            receiver_contact_states=self.receiver_contact_states,
            robot_contact_stop_names=[n for n in self.names if n not in (TARGET, RECEIVER)],
            strict_contact_diagnostic=dict(profile=STRICT_PROFILE, **strict))
        return primary


def enrich_pairs(pairs, meta, body_name, *, root_body_id, sim_time_s, step, pad_names):
    """S4 contact provenance with the S8 receiver ID, without changing old globals."""
    index = {n:i for i,n in enumerate(meta['geom_names'])}
    rows = []
    for pair in pairs:
        if pair['object'] != RECEIVER:
            raise ValueError('Only the declared cabinet receiver can be contact-exempt')
        rg, og = index[pair['robot_geom']], index[pair['object_geom']]
        rb, ob = int(meta['geom_body_ids'][rg]), int(meta['geom_body_ids'][og])
        name = pair['robot_geom']
        part = ('left_pad' if name in pad_names['left'] else 'right_pad'
                if name in pad_names['right'] else 'other_finger' if 'finger' in name
                else 'hand' if 'hand' in name else 'arm')
        rows.append(dict(pair, robot_geom_id=rg, receiver_geom_id=og,
            robot_body_id=rb, robot_body_name=body_name(rb), robot_part=part,
            receiver_contact_body_id=ob, receiver_contact_body_name=body_name(ob),
            receiver_root_body_id=int(root_body_id), receiver_root_body_name=body_name(root_body_id),
            signed_distance_m=float(pair['distance_m']),
            penetration_depth_m=max(0., -float(pair['distance_m'])),
            post_settle_control_step=int(step), sim_time_s=float(sim_time_s),
            primary_contact_stop=False))
    return rows


class CabinetMonitor(NonTargetMonitor):
    def __init__(self, env, target=TARGET):
        super().__init__(env, target)
        goal = next(g for g in self.d.parsed_problem['goal_state'] if g[1] == TARGET)
        if goal != ['on', TARGET, REGION] and goal != ['On', TARGET, REGION]:
            raise ValueError('S8 requires the public cabinet TOP_SIDE goal')
        if self.protection.receiver != RECEIVER:
            raise ValueError('Goal resolves to a different receiver')
        self.protection = CabinetProtection(self.names,
            self.d.sim.data.body_xpos[self.ids], self.d.sim.data.body_xquat[self.ids])
        self.receiver_root_id = self.d.obj_body_id[RECEIVER]
        self.first_receiver_contact = None

    def observe(self, action=None):
        state, target = super().observe(action)
        state['receiver_contacts'] = enrich_pairs(state['receiver_contacts'], self.meta,
            self.d.sim.model.body_id2name, root_body_id=self.receiver_root_id,
            sim_time_s=float(self.d.sim.data.time), step=self.step, pad_names=self.pad_names)
        if state['receiver_contacts'] and self.first_receiver_contact is None:
            self.first_receiver_contact = dict(step=self.step,
                sim_time_s=float(self.d.sim.data.time), pairs=copy.deepcopy(state['receiver_contacts']))
        state['first_receiver_contact'] = copy.deepcopy(self.first_receiver_contact)
        return state, target
