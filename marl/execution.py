import heapq

from configs import environment_config as env_cfg
from radar.models import RadarState


MOVEMENT_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
    4: (0, 0),
}


def project_action_position(env, agent, action):
    """Project a movement action without changing the environment."""

    current_x = agent.state.position.x
    current_y = agent.state.position.y
    delta_x, delta_y = MOVEMENT_DELTAS.get(
        action,
        (0, 0),
    )
    candidate_x = current_x + delta_x
    candidate_y = current_y + delta_y

    if not env._is_inside_grid(
        candidate_x,
        candidate_y,
    ):
        return current_x, current_y

    if not env.terrain.is_passable(
        candidate_x,
        candidate_y,
    ):
        return current_x, current_y

    return candidate_x, candidate_y


def get_hunter_expert_action(env):
    """Return the first step of the lowest-risk route to the target."""

    hunter_position = (
        env.hunter.state.position.x,
        env.hunter.state.position.y,
    )
    target_position = (
        env.strike_point.x,
        env.strike_point.y,
    )

    return _safest_path_action(
        env,
        hunter_position,
        target_position,
    )


def _radar_cell_risk(env, position):
    """Return a route cost that strongly prefers low-exposure cells."""

    risk = 0.0

    for radar in env.radar_system.radars:
        if not radar.active:
            continue

        radar_position = (
            radar.position.x,
            radar.position.y,
        )
        distance = env._dist(
            *position,
            *radar_position,
        )

        if position == radar_position:
            risk += 1000.0

        # A timer of one expires at the start of the next environment step.
        effectively_suppressed = (
            radar.suppression_timer > 1
        )

        if (
            not effectively_suppressed
            and distance <= radar.detection_range
        ):
            risk += 25.0 * (
                radar.detection_range
                - distance
                + 1.0
            )

    return risk


def _safest_path_action(env, start, target):
    """Find a terrain-valid, radar-risk-aware route with A* search."""

    if start == target:
        return env.ACTION_STAY

    action_order = (
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    )
    queue = [(0.0, 0.0, start, env.ACTION_STAY)]
    best_cost = {start: 0.0}

    while queue:
        _, path_cost, position, first_action = (
            heapq.heappop(queue)
        )

        if path_cost != best_cost.get(position):
            continue

        if position == target:
            return first_action

        for action in action_order:
            delta_x, delta_y = MOVEMENT_DELTAS[action]
            candidate = (
                position[0] + delta_x,
                position[1] + delta_y,
            )

            if not env._is_inside_grid(*candidate):
                continue

            if not env.terrain.is_passable(*candidate):
                continue

            candidate_cost = (
                path_cost
                + 1.0
                + _radar_cell_risk(env, candidate)
            )

            if candidate_cost >= best_cost.get(
                candidate,
                float("inf"),
            ):
                continue

            best_cost[candidate] = candidate_cost
            initial_action = (
                action
                if first_action == env.ACTION_STAY
                else first_action
            )
            heuristic = env._dist(
                *candidate,
                *target,
            )
            heapq.heappush(
                queue,
                (
                    candidate_cost + heuristic,
                    candidate_cost,
                    candidate,
                    initial_action,
                ),
            )

    return env.ACTION_STAY


def _nearest_jammable_radar(env):
    """Return the radar that suppression would target on the next step."""

    nearest = None
    nearest_distance = float("inf")

    for radar in env.radar_system.radars:
        if not radar.active:
            continue

        # advance_time() runs before the action, so timer=1 is available.
        if radar.suppression_timer > 1:
            continue

        distance = env.radar_system.distance(
            radar.position,
            env.scout.state.position,
        )

        if (
            distance <= env_cfg.JAM_RANGE
            and distance < nearest_distance
        ):
            nearest = radar
            nearest_distance = distance

    return nearest


def get_tactical_expert_actions(env):
    """Plan coordinated Scout jamming and Hunter movement actions."""

    scout_position = (
        env.scout.state.position.x,
        env.scout.state.position.y,
    )
    hunter_position = (
        env.hunter.state.position.x,
        env.hunter.state.position.y,
    )
    target_position = (
        env.strike_point.x,
        env.strike_point.y,
    )

    hunter_action = get_hunter_expert_action(env)
    projected_hunter_position = project_action_position(
        env,
        env.hunter,
        hunter_action,
    )
    scout_action = _safest_path_action(
        env,
        scout_position,
        target_position,
    )
    projected_scout_position = project_action_position(
        env,
        env.scout,
        scout_action,
    )

    jam_target = _nearest_jammable_radar(env)

    if jam_target is not None:
        radar_position = (
            jam_target.position.x,
            jam_target.position.y,
        )
        scout_track = jam_target.state_for(
            env.scout.state.aircraft_id
        )
        hunter_track = jam_target.state_for(
            env.hunter.state.aircraft_id
        )
        radar_threatens_route = (
            env._dist(
                *projected_scout_position,
                *radar_position,
            )
            <= jam_target.detection_range
            or env._dist(
                *projected_hunter_position,
                *radar_position,
            )
            <= jam_target.detection_range
            or scout_track != RadarState.SAFE
            or hunter_track != RadarState.SAFE
        )

        if radar_threatens_route:
            scout_action = env.ACTION_JAM_SUPPRESS
            projected_scout_position = scout_position

    hunter_enters_lethal_zone = False

    for radar in env.radar_system.radars:
        if not radar.active:
            continue

        radar_position = (
            radar.position.x,
            radar.position.y,
        )
        effectively_suppressed = (
            radar.suppression_timer > 1
        )
        suppressed_this_step = (
            scout_action == env.ACTION_JAM_SUPPRESS
            and radar is jam_target
        )

        if (
            not effectively_suppressed
            and not suppressed_this_step
            and env._dist(
                *projected_hunter_position,
                *radar_position,
            )
            <= radar.detection_range * 0.5
        ):
            hunter_enters_lethal_zone = True
            break

    escort_breaks = (
        env._dist(
            *projected_scout_position,
            *projected_hunter_position,
        )
        > env_cfg.ESCORT_RADIUS
    )

    if escort_breaks or hunter_enters_lethal_zone:
        candidates = []

        for action in MOVEMENT_DELTAS:
            candidate = project_action_position(
                env,
                env.hunter,
                action,
            )

            if (
                env._dist(
                    *projected_scout_position,
                    *candidate,
                )
                > env_cfg.ESCORT_RADIUS
            ):
                continue

            action_cost = (
                _radar_cell_risk(env, candidate)
                + env._dist(
                    *candidate,
                    *target_position,
                )
            )

            if action == env.ACTION_STAY:
                action_cost += 2.0

            candidates.append((action_cost, action))

        hunter_action = (
            min(candidates)[1]
            if candidates
            else env.ACTION_STAY
        )

    return scout_action, hunter_action


def apply_scout_jamming_guard(
    env,
    scout_action,
):
    """Keep Scout moving unless coordinated protection needs jamming."""

    expert_scout_action, expert_hunter_action = (
        get_tactical_expert_actions(env)
    )

    if expert_scout_action == env.ACTION_JAM_SUPPRESS:
        return env.ACTION_JAM_SUPPRESS

    # On an unfamiliar map the learned actor can otherwise repeat an
    # unnecessary jamming action forever. The tactical planner already
    # checks radar range, tracks and the projected route, so movement is
    # the safe fallback when it sees no suppression requirement.
    if scout_action in (
        env.ACTION_JAM_SUPPRESS,
        env.ACTION_JAM_DECEIVE,
    ):
        return expert_scout_action

    proposed_position = project_action_position(
        env,
        env.scout,
        scout_action,
    )
    expert_position = project_action_position(
        env,
        env.scout,
        expert_scout_action,
    )
    expert_hunter_position = project_action_position(
        env,
        env.hunter,
        expert_hunter_action,
    )
    target_position = (
        env.strike_point.x,
        env.strike_point.y,
    )

    expert_keeps_escort = (
        env._dist(
            *expert_position,
            *expert_hunter_position,
        )
        <= env_cfg.ESCORT_RADIUS
    )

    if not expert_keeps_escort:
        return scout_action

    proposed_distance = env._dist(
        *proposed_position,
        *target_position,
    )
    expert_distance = env._dist(
        *expert_position,
        *target_position,
    )
    proposed_risk = _radar_cell_risk(
        env,
        proposed_position,
    )
    expert_risk = _radar_cell_risk(
        env,
        expert_position,
    )
    current_position = (
        env.scout.state.position.x,
        env.scout.state.position.y,
    )
    proposed_makes_no_progress = (
        proposed_position == current_position
        and expert_position != current_position
    )

    if (
        proposed_makes_no_progress
        or proposed_risk > expert_risk
        or expert_distance < proposed_distance
    ):
        return expert_scout_action

    return scout_action


def apply_hunter_progress_guard(
    env,
    scout_action,
    hunter_action,
):
    """Reject unsafe or regressive Hunter motion.

    HAPPO remains responsible for the proposed action. The coordinated
    planner is used only as a fallback when its action keeps the formation
    intact and is safer than the proposal or makes strictly better progress.
    """

    _, expert_action = get_tactical_expert_actions(env)
    proposed_position = project_action_position(
        env,
        env.hunter,
        hunter_action,
    )
    expert_position = project_action_position(
        env,
        env.hunter,
        expert_action,
    )
    target_position = (
        env.strike_point.x,
        env.strike_point.y,
    )

    proposed_distance = env._dist(
        *proposed_position,
        *target_position,
    )
    expert_distance = env._dist(
        *expert_position,
        *target_position,
    )

    projected_scout_position = project_action_position(
        env,
        env.scout,
        scout_action,
    )
    projected_escort_distance = env._dist(
        *projected_scout_position,
        *expert_position,
    )

    if projected_escort_distance > env_cfg.ESCORT_RADIUS:
        return hunter_action

    proposed_escort_distance = env._dist(
        *projected_scout_position,
        *proposed_position,
    )
    proposed_risk = _radar_cell_risk(
        env,
        proposed_position,
    )
    expert_risk = _radar_cell_risk(
        env,
        expert_position,
    )

    if (
        proposed_escort_distance > env_cfg.ESCORT_RADIUS
        or proposed_risk > expert_risk
    ):
        return expert_action

    if expert_distance >= proposed_distance:
        return hunter_action

    return expert_action


def select_guarded_dqn_actions(env, scout_action):
    """Coordinate a DQN Scout with the tactical Hunter on custom maps.

    The DQN still proposes the Scout action. The guards only replace an
    unsafe/no-progress proposal and provide the Hunter with an escort-aware
    action instead of the environment's target-only heuristic.
    """

    guarded_scout_action = apply_scout_jamming_guard(
        env,
        scout_action,
    )
    _, expert_hunter_action = get_tactical_expert_actions(env)
    guarded_hunter_action = apply_hunter_progress_guard(
        env,
        guarded_scout_action,
        expert_hunter_action,
    )

    return guarded_scout_action, guarded_hunter_action
