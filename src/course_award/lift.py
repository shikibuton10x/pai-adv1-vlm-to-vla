"""Ground-truth scoring only. Never include these object coordinates in prompts."""
import numpy as np


class LiftTracker:
    def __init__(self):
        self.streak=0
        self.first_success_step=None
        self.last_step=None

    def update(self, *, rise_m, left, right, support_contacts, step):
        if type(step) is not int or step<0 or self.last_step is not None and step<=self.last_step:
            raise ValueError('Lift evidence must advance once per control state')
        self.last_step=step
        qualifies=bool(rise_m>=.020 and left and right and not support_contacts)
        self.streak=self.streak+1 if qualifies else 0
        if self.streak>=10 and self.first_success_step is None:
            self.first_success_step=step
        return dict(rise_m=float(rise_m),left_fingerpad=bool(left),right_fingerpad=bool(right),
            support_contacts=support_contacts,qualifying_streak=self.streak,
            lift_confirmed=self.first_success_step is not None,
            first_lift_step=self.first_success_step,measurement='target root-body world z, not EEF or COM')


class LiftMonitor:
    def __init__(self, env, target='wine_bottle_1'):
        self.d=env.env
        self.target=target
        self.root=self.d.obj_body_id[target]
        self.z0=float(self.d.sim.data.body_xpos[self.root,2])
        model=self.d.sim.model
        self.names=list(model.geom_names)
        desc={self.root}
        for i,parent in enumerate(model.body_parentid):
            if int(parent) in desc:desc.add(i)
        self.target_geoms={i for i,b in enumerate(model.geom_bodyid) if int(b) in desc}
        gripper=self.d.robots[0].gripper
        self.left={model.geom_name2id(n) for n in gripper.important_geoms['left_fingerpad']}
        self.right={model.geom_name2id(n) for n in gripper.important_geoms['right_fingerpad']}
        self.tracker=LiftTracker()

    def observe(self, step):
        others=[]
        for c in self.d.sim.data.contact[:self.d.sim.data.ncon]:
            if float(c.dist)>0:continue
            a,b=int(c.geom1),int(c.geom2)
            if b in self.target_geoms:a,b=b,a
            if a in self.target_geoms and b not in self.target_geoms:
                others.append(b)
        supports=sorted({self.names[i] for i in others if not self.names[i].startswith(('robot0_','gripper0_'))})
        evidence=self.tracker.update(rise_m=float(self.d.sim.data.body_xpos[self.root,2])-self.z0,
            left=bool(self.left.intersection(others)),right=bool(self.right.intersection(others)),
            support_contacts=supports,step=step)
        evidence.update(target=self.target,reference_body_z_m=self.z0,
            target_contact_other_geoms=sorted({self.names[i] for i in others}),
            sampling='20Hz signed-distance<=0 control states; substeps not inspected')
        return evidence
