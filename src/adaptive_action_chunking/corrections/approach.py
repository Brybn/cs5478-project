"""Optional lower-clearance approach; the original pickup teacher stays unchanged."""

import numpy as np

from .pickup import PickupTeacher, PICKUP_CONTROLLER
from .collect import cartesian_action


APPROACH_CONTROLLER = {
    "clearance_m": .03, "min_open_width_m": .06,
    "max_xy_m": .06, "min_height_m": -.01, "max_height_m": .06,
}


class ApproachTeacher(PickupTeacher):
    """Align at the current height when clear; otherwise raise only to 3 cm."""

    def __init__(self, state=None, reset_cube_z=.02):
        super().__init__(state, reset_cube_z)
        if state is None or self.phase != "retract":
            return
        tcp, cube = np.asarray(state[19:22]), np.asarray(state[29:32])
        settings = APPROACH_CONTROLLER
        if (state[18] <= .5 and state[7:9].sum() >= settings["min_open_width_m"]
                and .015 <= cube[2] <= reset_cube_z+.015
                and np.linalg.norm(tcp[:2]-cube[:2]) <= settings["max_xy_m"]
                and settings["min_height_m"] <= tcp[2]-cube[2] <= settings["max_height_m"]):
            self.approach_z = max(float(tcp[2]), float(cube[2])+settings["clearance_m"])
            self._phase("approach")

    def action(self, env, state):
        if self.phase == "approach":
            if self.phase_steps >= PICKUP_CONTROLLER["phase_timeout_steps"]:
                self._retry()
            else:
                tcp, cube = np.asarray(state[19:22]), np.asarray(state[29:32])
                target = np.array([*cube[:2], self.approach_z])
                if tcp[2] < self.approach_z-.003:
                    target[:2] = tcp[:2]  # Clear the cube before moving sideways.
                elif np.linalg.norm(tcp[:2]-cube[:2]) <= .003:
                    self._phase("descend")
                if self.phase == "approach":
                    action = cartesian_action(env, target-tcp, 1.,
                        max_step_m=PICKUP_CONTROLLER["travel_step_m"], deadband_m=.001,
                        gain=PICKUP_CONTROLLER["gain"])
                    self.steps += 1
                    self.phase_steps += 1
                    return action
        return super().action(env, state)
