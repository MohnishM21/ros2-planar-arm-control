# Controller state snapshot, published alongside /joint_states.
std_msgs/Header header

string state                    # IDLE | MOVING | STOPPED
string control_mode             # position | velocity | effort
string phase                    # pick-and-place phase, empty when not running

# Reference vs. measured joint positions (rad).
float64[] desired_position
float64[] tracking_error        # desired - measured
float64 trajectory_progress     # 0..1 of the active segment

geometry_msgs/Point end_effector
geometry_msgs/Point requested_target   # what the operator asked for
geometry_msgs/Point active_target      # what the planner is actually driving to
bool target_projected                  # true if requested target was unreachable

# Simulated gripper / payload.
bool gripper_closed
bool has_object
geometry_msgs/Point object_position

# Control-loop timing health.
float64 control_period_mean_ms
float64 control_jitter_max_ms

string message
