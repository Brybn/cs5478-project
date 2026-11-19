"""Continuous verified lift and gradual transport commands for the recovery teacher."""

import numpy as np

from .approach import ApproachTeacher
from .pickup import PICKUP_CONTROLLER


SMOOTH_CONTROLLER = {"max_step_m": .03, "max_step_change_m": .004}


class SmoothPickupTeacher(ApproachTeacher):
    """Retain grasp checks while limiting changes in Cartesian lift/place commands."""

    def __init__(self, state=None, reset_cube_z=.02):
        super().__init__(state, reset_cube_z)
        self.previous_delta = np.zeros(3, dtype=np.float32)

    def _limit_delta(self, desired, remaining_distance=None):
        delta = np.asarray(desired, dtype=np.float32).copy()
        limit = SMOOTH_CONTROLLER["max_step_m"]
        if remaining_distance is not None:
            change = SMOOTH_CONTROLLER["max_step_change_m"]
            # Leave room for one command and subsequent braking to the goal.
            limit = min(limit, np.sqrt(change*change+2*change*remaining_distance)-change)
        delta *= min(1., limit/max(float(np.linalg.norm(delta)), 1e-12))
        change = delta-self.previous_delta
        change *= min(1., SMOOTH_CONTROLLER["max_step_change_m"]/max(float(np.linalg.norm(change)), 1e-12))
        self.previous_delta = self.previous_delta+change
        return self.previous_delta.copy()

    def action(self, env, state):
        if self.phase == "lift" and state[18] > .5 and not self.verified and self.target is not None:
            # Continue rising during verification instead of braking at the first lift point.
            self.target[2] = max(float(self.target[2]), float(state[21])+PICKUP_CONTROLLER["lift_m"])
        action = super().action(env, state)
        if self.phase not in ("lift", "place") or self.failed:
            self.previous_delta[:] = 0
            return action
        config = env.unwrapped.agent.controller.controllers["arm"].config
        if config.frame != "root_translation" or not config.use_delta or config.use_target:
            raise ValueError("Smooth teacher requires instantaneous root-frame delta-position control")
        low = np.broadcast_to(np.asarray(config.pos_lower), (3,))
        high = np.broadcast_to(np.asarray(config.pos_upper), (3,))
        desired = (action[:3]+1)*(high-low)/2+low if config.normalize_action else action[:3]
        remaining = float(np.linalg.norm(state[26:29]-state[29:32])) if self.phase == "place" else None
        delta = self._limit_delta(desired, remaining)
        action = action.copy()
        action[:3] = 2*(delta-low)/(high-low)-1 if config.normalize_action else delta
        return action


class DescentTeacher(SmoothPickupTeacher):
    """Descend immediately when an open takeover is already aligned above the cube."""

    def __init__(self, state=None, reset_cube_z=.02):
        super().__init__(state, reset_cube_z)
        if (state is not None and self.phase == "approach"
                and np.linalg.norm(state[19:21]-state[29:31]) <= .003
                and state[21] >= state[31]+PICKUP_CONTROLLER["grasp_offset_m"]):
            self._phase("descend")


class LevelApproachTeacher(DescentTeacher):
    """Align near the cube at the current height, then descend without a clearance rise."""

    def __init__(self, state=None, reset_cube_z=.02):
        super().__init__(state, reset_cube_z)
        self.level_approach = (state is not None and self.phase == "approach"
            and np.linalg.norm(state[19:21]-state[29:31]) <= .015
            and state[21]-state[31] >= .011)
        if self.level_approach:
            self.approach_z = float(state[21])

    def action(self, env, state):
        if self.level_approach and self.phase == "approach":
            # Preserve ongoing downward motion instead of restoring takeover height.
            self.approach_z = min(self.approach_z, float(state[21]))
        return super().action(env, state)
