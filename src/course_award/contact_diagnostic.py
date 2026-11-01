"""Prospective post-quality diagnostic gating; historical clean is never reset."""
import copy

DIAGNOSTIC_SYSTEM = '''This run is a post-quality-ineligibility grasp connection diagnostic.
The original quality failure remains permanent; do not claim quality success.
robot_contact reports ONLY whether your own hand/fingers contact something, with
no counterpart identity or object coordinates. Before the first close command,
hand contact interrupts move/rotate after a control step for fresh observation.
Open/close/hold still execute their bounded steps; finger contact after closing
is not by itself a motion interrupt. Close is still not proof of grasp or lift.'''


def teacher_motion_feedback(motion):
    """Keep internal contact classifications out of the teacher-side feedback."""
    if motion is None:return None
    public=copy.deepcopy(motion)
    if public.get('reason')=='TARGET_HAND_CONTACT_REOBSERVE':
        public['reason']='HAND_CONTACT_REOBSERVE'
    return public


def robot_part(name):
    if name.startswith(('gripper0_','robot0_')):
        if 'finger' in name:return 'finger'
        if 'hand' in name:return 'hand'
    return None


def self_contact_flags(domain):
    """Robot-to-environment contact only, never robot self-collision or counterpart ID."""
    names=list(domain.sim.model.geom_names)
    pairs=[]
    for contact in domain.sim.data.contact[:domain.sim.data.ncon]:
        if float(contact.dist)>0:continue
        pairs.append(dict(geom1=names[int(contact.geom1)],geom2=names[int(contact.geom2)]))
    return environment_contact_flags(pairs)


def environment_contact_flags(pairs):
    """Same boundary for current sim contacts and saved robot-contact preview rows."""
    flags=dict(any=False,hand=False,finger=False)
    for pair in pairs:
        if pair.get('distance_m',0)>0:continue
        a,b=pair['geom1'],pair['geom2']
        ar=a.startswith(('gripper0_','robot0_'));br=b.startswith(('gripper0_','robot0_'))
        if ar==br:continue  # both robot (self-collision) or neither robot
        name=a if ar else b
        flags['any']=True
        part=robot_part(name)
        if part:flags[part]=True
    return flags


def diagnostic_gate(whole_status, *, target, close_proxy_seen, operation):
    """Only relax the old target-only contact stop; all other failures persist.

    Calls follow each physical control step. A pre-close hand-touching move
    returns immediately instead of executing all20 steps while contact persists.
    """
    if whole_status['first_protection_violation'] is not None:
        return 'QUALITY_STOP'
    history=whole_status.get('first_contact_by_object',{})
    if any(name!=target for name in history):
        return 'QUALITY_STOP'
    contacts=whole_status.get('contacts',[])
    if any(row['object']!=target for row in contacts):
        return 'QUALITY_STOP'
    if contacts and not close_proxy_seen:
        if operation in ('move_delta','rotate_delta'):
            if all(robot_part(row['robot_geom'])=='hand' for row in contacts):
                return 'TARGET_HAND_CONTACT_REOBSERVE'
            # Unspecified pre-close contact is not covered by this exception.
            return 'QUALITY_STOP'
        if operation not in ('open','close','hold'):
            return 'QUALITY_STOP'
    return None
