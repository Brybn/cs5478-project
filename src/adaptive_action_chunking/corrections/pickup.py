"""Expert pickup retries using the policy's existing XYZ/gripper actions."""

import numpy as np

from .collect import cartesian_action


PICKUP_CONTROLLER = {
    "clearance_m": .06, "grasp_offset_m": .002, "lift_m": .04,
    "verify_rise_m": .025, "verify_steps": 5, "open_steps": 4, "close_steps": 4,
    "min_grasp_width_m": .025, "max_grasp_xy_m": .015,
    "max_unheld_cube_travel_m": .03,
    "gain": 1.0, "travel_step_m": .03, "descend_step_m": .02,
    "lift_step_m": .015, "place_step_m": .05,
    "max_attempts": 3, "phase_timeout_steps": 35, "max_steps": 180,
    "accept_distance_m": .010, "success_hold_steps": 10, "min_steps": 20,
}
PICKUP_GROUPS = ("retry", "lift", "place", "successful")


def choose_pickup_takeover(source, group=None):
    """Cover retry, held-lift and placement states, plus complete successful runs."""
    states = np.asarray(source["states"])
    if group == "successful":
        return (0, "successful_disturbed_sequence") if source["success_at_end"] else (None, None)
    if source["success_at_end"]:
        return None, None
    actions = np.asarray(source.get("actions", []))
    lifted = (states[:, 18] > .5) & (states[:, 31] >= states[0, 31] + PICKUP_CONTROLLER["verify_rise_m"])
    source_lifted = bool(np.any(np.convolve(lifted.astype(int), np.ones(5, dtype=int), mode="valid") == 5))
    if (group == "lift" and source_lifted) or (group == "place" and not source_lifted):
        return None, None
    fallback = (None, None)
    for step in range(30, len(states)):
        tail = states[step - 4:step + 1]
        cube = states[step, 29:32]
        if np.any(np.abs(cube[:2]) > .3) or cube[2] < .015:
            continue
        held = np.all(tail[:, 18] > .5)
        raised = np.all(tail[:, 31] >= states[0, 31] + PICKUP_CONTROLLER["verify_rise_m"])
        if held and group == "place":
            return step, "placement_intervention"
        if held and group in (None, "lift") and step >= 40 and not raised:
            if cube[2] < states[0, 31] + .025 and cube[2] - tail[0, 31] <= .005:
                return step, "held_without_lift"
        if held and raised and group in (None, "place"):
            return step, "placement_intervention"
        if group not in (None, "retry") or np.any(tail[:, 18] > .5) or cube[2] > .08:
            continue
        if np.linalg.norm(np.diff(tail[:, 29:32], axis=0), axis=1).sum() > PICKUP_CONTROLLER["max_unheld_cube_travel_m"]:
            continue
        near = (np.linalg.norm(states[step, 19:21] - cube[:2]) <= .015
                and abs(states[step, 21] - cube[2]) <= .015)
        if (near and len(actions) > step and actions[step - 1, 3] < -.5 and actions[step, 3] > .5):
            return step, "premature_reopening"
        if np.any(states[:step, 18] > .5):
            return step, "lost_grasp"
        empty_close = (step >= 40 and len(actions) >= step
                       and np.all(actions[step - 3:step, 3] < -.5)
                       and states[step, 7:9].sum() < .025)
        if (empty_close or step >= 50) and fallback[0] is None:
            fallback = (step, "missed_grasp")
    return fallback


class PickupTeacher:
    """Open, retract, align, descend, close, verify a lift, then place."""

    def __init__(self, state=None, reset_cube_z=.02):
        self.phase = "retract"
        self.phase_steps = self.steps = 0
        self.attempts = 1
        self.target = None
        self.lift_start_z = None
        self.verified = False
        self.grasp_streak = 0
        self.failed = False
        self.events = []
        if state is not None:
            tcp, cube = np.asarray(state[19:22]), np.asarray(state[29:32])
            seated = (state[7:9].sum() >= PICKUP_CONTROLLER["min_grasp_width_m"]
                      and np.linalg.norm(tcp[:2] - cube[:2]) <= PICKUP_CONTROLLER["max_grasp_xy_m"])
            if state[18] > .5 and (seated or cube[2] >= reset_cube_z + .025):
                self.lift_start_z = float(reset_cube_z if cube[2] >= reset_cube_z + .025 else cube[2])
                self._phase("place" if cube[2] >= reset_cube_z + .025 else "lift")
                if self.phase == "lift":
                    self.target = tcp + [0, 0, PICKUP_CONTROLLER["lift_m"]]
            elif state[18] <= .5 and (np.linalg.norm(tcp[:2] - cube[:2]) <= .006
                  and abs(tcp[2] - cube[2] - PICKUP_CONTROLLER["grasp_offset_m"]) <= .008):
                self._phase("close")

    def _phase(self, name):
        self.events.append({"step": self.steps, "phase": name})
        self.phase, self.phase_steps, self.target = name, 0, None

    def _retry(self):
        self.verified = False
        self.grasp_streak = 0
        if self.attempts >= PICKUP_CONTROLLER["max_attempts"]:
            self.failed = True
        else:
            self.attempts += 1
            self._phase("retract")

    def action(self, env, state):
        tcp, cube = np.asarray(state[19:22]), np.asarray(state[29:32])
        held = bool(state[18] > .5)
        if self.phase_steps >= PICKUP_CONTROLLER["phase_timeout_steps"] and self.phase != "place":
            self._retry()
        if self.phase == "place" and not held:
            self._retry()

        # Transitions use the observed result of preceding executed actions.
        if self.phase == "retract":
            if self.target is None:
                self.target = tcp.copy()
                self.target[2] = max(float(tcp[2]), float(cube[2]) + PICKUP_CONTROLLER["clearance_m"])
            if self.phase_steps >= PICKUP_CONTROLLER["open_steps"] and np.linalg.norm(tcp - self.target) <= .004:
                self._phase("align")
        if self.phase == "align":
            self.target = cube + [0, 0, PICKUP_CONTROLLER["clearance_m"]]
            if np.linalg.norm(tcp - self.target) <= .003:
                self._phase("descend")
        if self.phase == "descend":
            self.target = cube + [0, 0, PICKUP_CONTROLLER["grasp_offset_m"]]
            if np.linalg.norm(tcp - self.target) <= .002:
                self._phase("close")
        if self.phase == "close":
            self.target = cube + [0, 0, PICKUP_CONTROLLER["grasp_offset_m"]]
            if self.phase_steps >= PICKUP_CONTROLLER["close_steps"] and (held or self.phase_steps >= 6):
                self.lift_start_z = float(cube[2])
                self._phase("lift")
                self.target = tcp + [0, 0, PICKUP_CONTROLLER["lift_m"]]
        if self.phase in ("lift", "place") and not self.verified:
            risen = cube[2] >= self.lift_start_z + PICKUP_CONTROLLER["verify_rise_m"]
            self.grasp_streak = self.grasp_streak + 1 if held and risen else 0
            if self.grasp_streak >= PICKUP_CONTROLLER["verify_steps"]:
                self.verified = True
                self.events.append({"step": self.steps, "verified_lift": True,
                                    "start_cube_z": self.lift_start_z, "cube_z": float(cube[2])})
                self._phase("place")
            elif self.phase == "lift" and self.phase_steps >= 12 and not held:
                self._retry()

        gripper = 1.0 if self.phase in ("retract", "align", "descend") else -1.0
        if self.phase == "place":
            action = cartesian_action(env, state[26:29] - cube, max_step_m=PICKUP_CONTROLLER["place_step_m"],
                                      gain=PICKUP_CONTROLLER["gain"])
        else:
            if self.target is None:  # A failed lift can start a fresh retract here.
                self.target = tcp.copy()
                self.target[2] = max(float(tcp[2]), float(cube[2]) + PICKUP_CONTROLLER["clearance_m"])
            limit = PICKUP_CONTROLLER.get(self.phase + "_step_m", PICKUP_CONTROLLER["travel_step_m"])
            action = cartesian_action(env, self.target - tcp, gripper, max_step_m=limit, deadband_m=.001,
                                      gain=PICKUP_CONTROLLER["gain"])
        self.steps += 1
        self.phase_steps += 1
        return action
