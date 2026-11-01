"""Versioned teacher contract: no target-coordinate oracle or legacy quality latch."""
from .gemini import build_request

SYSTEM='''S1_NONTARGET_V2: fresh wine-bottle grasp/lift; no rack placement.
The bottle is ALWAYS excluded from non-target preservation, open or closed.
Bottle hand/arm/finger contacts and movement are logged, not failures alone.
ALL other named objects/fixtures, including receiver rack, remain protected:
robot contact stops; translation limit1mm (receiver3mm), tilt3degrees.
Closing/opening NEVER disables these rules. Robot-table/workspace contacts and
substeps are not preservation-audited; still avoid collisions.
Close is not grasp evidence. Moving lift requires bottle root rise>=20mm, BOTH
target fingerpads and no nonrobot support for10 consecutive20Hz observations.
Ascent is allowed; this is NOT stationary0.5s holding. On confirmation the system
automatically holds10 control steps, retaining your LATEST gripper command;
it never forces close. Start+10 states must EACH satisfy rise20mm/both pads/no
support, and simulation duration>=0.5s. Failure ends HOLD_FAILED, no retry.
Non-target violation takes priority. Success requires preservation+lift+retention.
A new environment HAND contact false->true interrupts move/rotate for observation.
Known continuous contact alone does not interrupt again. Edge state updates on
EVERY step, including close/open/hold, independently of close commands.
Teacher380 steps + reserved automatic hold10 + preparation10 = maximum400.
There are at most20 new generation attempts. Inspect remaining-step feedback.
Images1/2/3 are CURRENT agent/wrist/world-fixed side RGB at one frozen state.
robot_card contains ONLY Panda/camera geometry: EEF/pad offset, jaw/approach
axes, camera poses/intrinsics and robot pixel projections (visibility uncertain).
Wrist image center is optical axis, not grip center. recent_operations contains
only this episode's last4 robot actions. No old success sequence/object coordinates.
Contact flags exclude robot self-collision and do not identify the counterpart.
World moves use metres; rotations use world axis-angle radians, left-multiplied.
Natural images: raw vertically flipped once, u right/v down. Agent XYZ axes50mm.
Return ONE JSON, no code/prose/unknown fields:
{"operation":"move_delta","delta_world_m":[0,0,0.02],"steps":20}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":10} / {"operation":"close","steps":10}
{"operation":"hold","steps":10} / {"operation":"stop"}.
Finite vectors; integer steps1..20. Norms above0.06m/0.35rad are direction-preserving
clipped. Actual movement may be smaller. Moves retain latest gripper command.
Near-closed gap without finger-environment contact: reposition from images or stop;
do not repeat open/close at the same pose. Never output object coordinates.'''


HAND_INTERRUPT_TEXT='''A new environment HAND contact false->true interrupts move/rotate for observation.
Known continuous contact alone does not interrupt again. Edge state updates on
EVERY step, including close/open/hold, independently of close commands.'''
ADVISORY_HAND_TEXT='''HAND contact onset is advisory/log-only: it never interrupts move/rotate.
Non-target violations still stop immediately. Edge state updates on EVERY step,
including close/open/hold, independently of close commands.'''
assert HAND_INTERRUPT_TEXT in SYSTEM
ADVISORY_SYSTEM=SYSTEM.replace('S1_NONTARGET_V2','S1_NONTARGET_V3_ADVISORY',1).replace(HAND_INTERRUPT_TEXT,ADVISORY_HAND_TEXT)


def request(payload,paths,*,advisory=False):
    value=build_request(payload,paths[:2],current_side_image_path=paths[2])
    value['systemInstruction']={'parts':[{'text':ADVISORY_SYSTEM if advisory else SYSTEM}]}
    return value


def payload(instruction,robot,flags,card,history,*,step,gripper,feedback,frame,protocol='S1_NONTARGET_V2'):
    return dict(protocol=protocol,instruction=instruction,
        current_test='non-target preservation + physical moving lift + automatic retention; no placement',
        robot=robot,robot_contact=flags,robot_card=card,recent_operations=history[-4:],
        control_step=step,remaining_teacher_steps=380-step,reserved_automatic_hold_steps=10,
        gripper_command=gripper,previous_rejection=feedback,frame_sequence=frame,
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_sideview_natural'])
