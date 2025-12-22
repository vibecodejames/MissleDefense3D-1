"""
Core data structures and math utilities for missile simulation.

Contains:
- Enums for guidance phases, seeker states, and intercept results
- Data classes for Aircraft, Seeker, TargetEstimator, GuidanceController
- Multi-target tracking structures (TargetTrack, InterceptorStatus, SAMBattery)
- Core math functions for intercept calculations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto


# =============================================================================
# GUIDANCE ENUMS
# =============================================================================

class GuidancePhase(Enum):
    """Multi-phase guidance states for SAM missiles."""
    BOOST = auto()      # Initial climb, fixed attitude toward predicted intercept
    MIDCOURSE = auto()  # Command guidance from ground radar, coarse PN
    TERMINAL = auto()   # Active seeker locked, true Proportional Navigation


class SeekerState(Enum):
    """Seeker head tracking states."""
    OFF = auto()        # Seeker not active (boost phase)
    SEARCHING = auto()  # Looking for target within acquisition cone
    LOCKED = auto()     # Tracking target within gimbal limits
    LOST = auto()       # Target left gimbal limits, attempting re-acquisition


class InterceptResult(Enum):
    """Result of an intercept attempt."""
    IN_PROGRESS = auto()  # Engagement still active
    HIT = auto()          # Target destroyed
    MISS = auto()         # Interceptor missed, target continues


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Aircraft:
    """Represents an aircraft/missile with position, velocity, and flight dynamics."""
    name: str
    position: np.ndarray      # [x, y] in meters
    velocity: np.ndarray      # [vx, vy] in meters/second
    max_speed: float          # m/s - maximum velocity magnitude

    # Flight dynamics parameters
    max_acceleration: float = 50.0    # m/s² - max thrust acceleration
    max_g: float = 30.0               # Maximum sustainable g-load for turns
    burn_time: float = 60.0           # seconds of powered flight (fuel)
    drag_coefficient: float = 0.001   # simplified drag factor (higher = more drag)
    is_missile: bool = True           # missiles affected by gravity, aircraft maintain altitude

    # State tracking (not set at init, updated during simulation)
    time_since_launch: float = 0.0    # seconds since launch
    is_burning: bool = True           # True if still in powered flight

    @property
    def speed(self) -> float:
        """Current speed (magnitude of velocity vector)."""
        return np.linalg.norm(self.velocity)

    @property
    def heading(self) -> float:
        """Current heading in degrees (0 = East, 90 = North)."""
        return np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))

    def position_at_time(self, t: float) -> np.ndarray:
        """Where will this aircraft be at time t (assuming constant velocity)?"""
        return self.position + self.velocity * t

    def __str__(self):
        burn_status = "BURN" if self.is_burning else "COAST"
        return (f"{self.name}: pos=({self.position[0]:.0f}, {self.position[1]:.0f})m, "
                f"speed={self.speed:.0f}m/s, heading={self.heading:.1f}°, {burn_status}")


@dataclass
class Seeker:
    """
    Models an active radar or IR seeker head.

    The seeker has:
    - Acquisition cone: Target must be within this cone to acquire lock
    - Gimbal limits: Maximum angle the seeker can track off-boresight
    - Lock-on time: Time target must be in cone before lock acquired
    - State machine: OFF -> SEARCHING -> LOCKED (or LOST)
    """

    # Seeker geometry (degrees)
    acquisition_cone: float = 30.0    # Half-angle for target acquisition
    gimbal_limit: float = 45.0        # Maximum off-boresight tracking angle

    # Seeker performance
    max_range: float = 20_000.0       # Maximum seeker lock range (meters)
    min_range: float = 100.0          # Minimum range (fusing distance)
    lock_time: float = 0.5            # Time to acquire lock (seconds)

    # State tracking
    state: SeekerState = field(default=SeekerState.OFF)
    target_bearing: float = 0.0       # Angle to target from boresight (degrees)
    time_in_cone: float = 0.0         # Time target has been in acquisition cone
    los_rate: float = 0.0             # Line-of-sight rate (rad/s)
    closing_velocity: float = 0.0     # Vc = -dR/dt (m/s, positive when closing)

    # Previous state for rate calculations
    _prev_los_angle: float = 0.0

    def update(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
               missile_heading: float, target_pos: np.ndarray,
               target_vel: np.ndarray, dt: float) -> bool:
        """
        Update seeker state based on current geometry.

        Returns:
            True if target is being tracked (locked)
        """
        # Calculate relative geometry
        to_target = target_pos - missile_pos
        range_to_target = np.linalg.norm(to_target)

        # Calculate bearing (angle from boresight to target)
        target_angle = np.degrees(np.arctan2(to_target[1], to_target[0]))
        self.target_bearing = self._normalize_angle(target_angle - missile_heading)

        # Calculate LOS rate and closing velocity
        self._update_los_rate(missile_pos, missile_vel, target_pos, target_vel, dt)
        self._update_closing_velocity(missile_pos, missile_vel, target_pos, target_vel)

        # State machine
        if self.state == SeekerState.OFF:
            return False

        elif self.state == SeekerState.SEARCHING:
            if self._is_in_acquisition_cone() and range_to_target <= self.max_range:
                self.time_in_cone += dt
                if self.time_in_cone >= self.lock_time:
                    self.state = SeekerState.LOCKED
                    return True
            else:
                self.time_in_cone = 0.0
            return False

        elif self.state == SeekerState.LOCKED:
            if not self._is_in_gimbal_limits():
                self.state = SeekerState.LOST
                self.time_in_cone = 0.0
                return False
            if range_to_target > self.max_range * 1.2:  # Hysteresis
                self.state = SeekerState.LOST
                self.time_in_cone = 0.0
                return False
            return True

        elif self.state == SeekerState.LOST:
            # Can re-acquire if back in cone
            if self._is_in_acquisition_cone() and range_to_target <= self.max_range:
                self.time_in_cone += dt
                if self.time_in_cone >= self.lock_time * 1.5:  # Longer re-acquire time
                    self.state = SeekerState.LOCKED
                    return True
            else:
                self.time_in_cone = 0.0
            return False

        return False

    def _update_los_rate(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
                         target_pos: np.ndarray, target_vel: np.ndarray, dt: float):
        """Calculate line-of-sight rate: dλ/dt"""
        R = target_pos - missile_pos
        V_rel = target_vel - missile_vel

        R_squared = np.dot(R, R)
        if R_squared < 100:  # Avoid singularity at close range
            self.los_rate = 0.0
            return

        # LOS rate = (Rx * Vy - Ry * Vx) / R²
        self.los_rate = (R[0] * V_rel[1] - R[1] * V_rel[0]) / R_squared

    def _update_closing_velocity(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
                                  target_pos: np.ndarray, target_vel: np.ndarray):
        """Calculate closing velocity: Vc = -dR/dt"""
        R = target_pos - missile_pos
        range_mag = np.linalg.norm(R)

        if range_mag < 1.0:
            self.closing_velocity = 0.0
            return

        R_unit = R / range_mag
        V_rel = missile_vel - target_vel  # Missile relative to target

        # Closing velocity = component of relative velocity toward target
        self.closing_velocity = np.dot(V_rel, R_unit)

    def _is_in_acquisition_cone(self) -> bool:
        """Check if target is within acquisition cone."""
        return abs(self.target_bearing) <= self.acquisition_cone

    def _is_in_gimbal_limits(self) -> bool:
        """Check if target is within gimbal tracking limits."""
        return abs(self.target_bearing) <= self.gimbal_limit

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] degrees."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def activate(self):
        """Activate seeker (start searching)."""
        if self.state == SeekerState.OFF:
            self.state = SeekerState.SEARCHING
            self.time_in_cone = 0.0


@dataclass
class TargetEstimator:
    """
    Kalman filter for target state estimation.

    Smooths noisy seeker measurements and predicts target position
    between updates. Uses a constant velocity motion model.

    State vector: [x, y, vx, vy]
    """

    # Process noise (target maneuver capability)
    process_noise: float = 50.0      # m/s² equivalent acceleration noise

    # Measurement noise (seeker accuracy)
    measurement_noise: float = 100.0  # meters position uncertainty

    # Internal state
    _initialized: bool = field(default=False, repr=False)
    _x: np.ndarray = field(default=None, repr=False)  # State estimate [x, y, vx, vy]
    _P: np.ndarray = field(default=None, repr=False)  # State covariance 4x4

    def predict(self, dt: float):
        """Predict step: propagate state forward using constant velocity model."""
        if not self._initialized:
            return

        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise covariance
        q = self.process_noise
        Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q**2

        # Predict state and covariance
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, measurement: np.ndarray):
        """Update step: incorporate position measurement from seeker."""
        if not self._initialized:
            # Initialize state from first measurement
            self._x = np.array([measurement[0], measurement[1], 0.0, 0.0])
            self._P = np.diag([self.measurement_noise**2, self.measurement_noise**2,
                               1000.0, 1000.0])
            self._initialized = True
            return

        # Measurement matrix (observe position only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise covariance
        R = np.eye(2) * self.measurement_noise**2

        # Innovation (measurement residual)
        y = measurement - H @ self._x

        # Innovation covariance
        S = H @ self._P @ H.T + R

        # Kalman gain
        K = self._P @ H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(4)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T

    def reset(self):
        """Reset the estimator (for new engagement)."""
        self._initialized = False
        self._x = None
        self._P = None

    @property
    def position(self) -> Optional[np.ndarray]:
        """Estimated target position [x, y]."""
        if not self._initialized:
            return None
        return self._x[:2].copy()

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Estimated target velocity [vx, vy]."""
        if not self._initialized:
            return None
        return self._x[2:].copy()

    @property
    def state(self) -> Optional[np.ndarray]:
        """Full state estimate [x, y, vx, vy]."""
        if not self._initialized:
            return None
        return self._x.copy()


@dataclass
class GuidanceController:
    """
    Multi-phase guidance controller.

    Manages transitions between guidance phases:
    - BOOST: Initial climb, no active guidance
    - MIDCOURSE: Command guidance from ground radar
    - TERMINAL: Active seeker with true Proportional Navigation
    """

    # Phase transition parameters
    boost_duration: float = 3.0          # Seconds of boost phase (no guidance)
    terminal_range: float = 15_000.0     # Range at which terminal guidance can begin (m)

    # Navigation constants
    N_midcourse: float = 3.0             # PN constant for midcourse (coarse)
    N_terminal: float = 4.0              # PN constant for terminal (fine)

    # Ground radar parameters (for midcourse)
    ground_radar_update_rate: float = 2.0  # Updates per second
    last_radar_update: float = 0.0

    # State tracking
    current_phase: GuidancePhase = field(default=GuidancePhase.BOOST)
    phase_start_time: float = 0.0

    # Stored midcourse aim point (updated by ground radar)
    midcourse_aim_point: np.ndarray = None

    def update_phase(self, time_since_launch: float,
                     range_to_target: float,
                     seeker: Seeker) -> GuidancePhase:
        """
        Determine and update current guidance phase.

        Phase transitions:
        - BOOST -> MIDCOURSE: After boost_duration seconds
        - MIDCOURSE -> TERMINAL: When seeker locks target
        - TERMINAL -> MIDCOURSE: If seeker loses lock (fallback)
        """
        old_phase = self.current_phase

        # BOOST -> MIDCOURSE: After boost duration
        if (self.current_phase == GuidancePhase.BOOST and
                time_since_launch >= self.boost_duration):
            self.current_phase = GuidancePhase.MIDCOURSE
            self.phase_start_time = time_since_launch

        # MIDCOURSE -> TERMINAL: When seeker locks target
        if (self.current_phase == GuidancePhase.MIDCOURSE and
                seeker.state == SeekerState.LOCKED):
            self.current_phase = GuidancePhase.TERMINAL
            self.phase_start_time = time_since_launch

        # TERMINAL -> MIDCOURSE (fallback): If seeker loses lock
        if (self.current_phase == GuidancePhase.TERMINAL and
                seeker.state == SeekerState.LOST):
            self.current_phase = GuidancePhase.MIDCOURSE
            self.phase_start_time = time_since_launch

        # Log phase transitions
        if self.current_phase != old_phase:
            print(f"  >>> GUIDANCE: {old_phase.name} -> {self.current_phase.name}")

        return self.current_phase

    def should_update_radar(self, current_time: float) -> bool:
        """Check if ground radar should send an update."""
        if current_time - self.last_radar_update >= 1.0 / self.ground_radar_update_rate:
            self.last_radar_update = current_time
            return True
        return False


# =============================================================================
# MULTI-TARGET TRACKING STRUCTURES
# =============================================================================

@dataclass
class TargetTrack:
    """
    Tracks a single incoming threat (cruise missile).

    Used in saturation attack scenarios to manage multiple targets
    with prioritization and engagement status.
    """
    target_id: int
    aircraft: Aircraft                    # The cruise missile being tracked
    status: str = "incoming"              # incoming, engaged, destroyed, leaked
    assigned_interceptor_id: Optional[int] = None
    priority: float = 0.0                 # Higher = more urgent (based on time-to-impact)
    time_to_impact: float = float('inf')  # Estimated seconds until city impact
    range_to_city: float = float('inf')   # Current distance to city (meters)
    path_history: List[np.ndarray] = field(default_factory=list)

    def record_position(self):
        """Record current position to path history."""
        self.path_history.append(self.aircraft.position.copy())


@dataclass
class InterceptorStatus:
    """
    Tracks a single SAM interceptor.

    Manages the lifecycle of an interceptor from ready → launched →
    in_flight → hit/miss/expended.
    """
    interceptor_id: int
    aircraft: Optional[Aircraft] = None   # None if not yet launched
    status: str = "ready"                 # ready, launching, in_flight, hit, miss, expended
    assigned_target_id: Optional[int] = None
    launch_time: Optional[float] = None

    # Guidance components (created on launch)
    seeker: Optional[Seeker] = None
    guidance: Optional[GuidanceController] = None
    target_estimator: Optional[TargetEstimator] = None

    path_history: List[np.ndarray] = field(default_factory=list)

    def record_position(self):
        """Record current position to path history."""
        if self.aircraft is not None:
            self.path_history.append(self.aircraft.position.copy())


@dataclass
class SAMBattery:
    """
    Represents a Surface-to-Air Missile battery with multiple launchers.

    Manages inventory, reload times, and launch coordination for
    defending against saturation attacks.
    """
    position: np.ndarray                  # Battery location [x, y]
    total_interceptors: int = 8           # Total missile inventory
    reload_time: float = 8.0              # Seconds between consecutive launches
    max_simultaneous: int = 4             # Max interceptors in flight at once
    pk_single: float = 0.85               # Single-shot probability of kill

    # State tracking
    last_launch_time: float = -100.0      # Time of last launch (for reload)
    interceptors_remaining: int = 8       # Current inventory
    interceptors_in_flight: int = 0       # Currently flying

    # Interceptor list (initialized on setup)
    interceptors: List[InterceptorStatus] = field(default_factory=list)

    def can_launch(self, current_time: float) -> bool:
        """Check if battery can launch a new interceptor."""
        if self.interceptors_remaining <= 0:
            return False
        if self.interceptors_in_flight >= self.max_simultaneous:
            return False
        if current_time - self.last_launch_time < self.reload_time:
            return False
        return True

    def time_until_ready(self, current_time: float) -> float:
        """Time until next launch is possible (0 if ready now)."""
        if self.interceptors_remaining <= 0:
            return float('inf')
        if self.interceptors_in_flight >= self.max_simultaneous:
            return float('inf')
        reload_remaining = self.reload_time - (current_time - self.last_launch_time)
        return max(0.0, reload_remaining)


# =============================================================================
# CORE MATH: INTERCEPT CALCULATIONS
# =============================================================================

def calculate_intercept_time(pursuer: Aircraft, target: Aircraft,
                             use_current_speed: bool = False) -> Optional[float]:
    """
    Calculate time for pursuer to intercept target (both moving at constant velocity).

    THE MATH:
    ---------
    We need to find time t where: pursuer_position(t) = target_position(t)

    But the pursuer can change direction! So we ask:
    "At what time t can the pursuer reach the target's future position?"

    This becomes: |target_pos + target_vel * t - pursuer_pos| = pursuer_speed * t

    Squaring both sides and expanding gives us a quadratic equation:
    at² + bt + c = 0

    where:
    - a = |target_vel|² - pursuer_speed²
    - b = 2 * (relative_pos · target_vel)
    - c = |relative_pos|²

    Returns None if intercept is impossible.
    """
    # Relative position: vector from pursuer to target
    rel_pos = target.position - pursuer.position

    # Pursuer speed - use current speed for better accuracy during acceleration
    if use_current_speed:
        v_p = pursuer.speed
    else:
        v_p = pursuer.max_speed

    # Target velocity
    v_t = target.velocity

    # Quadratic coefficients
    a = np.dot(v_t, v_t) - v_p**2
    b = 2 * np.dot(rel_pos, v_t)
    c = np.dot(rel_pos, rel_pos)

    # Solve quadratic: at² + bt + c = 0
    discriminant = b**2 - 4*a*c

    # CASE 1: No real solutions - intercept impossible
    if discriminant < 0:
        return None

    # CASE 2: a ≈ 0 (speeds are equal) - linear solution
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            return None  # No relative motion
        t = -c / b
        return t if t > 0 else None

    # CASE 3: Quadratic solutions
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    # We want the smallest POSITIVE time
    valid_times = [t for t in [t1, t2] if t > 0]

    if not valid_times:
        return None

    return min(valid_times)


def calculate_lead_angle(pursuer: Aircraft, target: Aircraft,
                         intercept_time: float) -> Tuple[float, np.ndarray]:
    """
    Calculate the heading the pursuer needs to intercept the target.

    Returns:
    - lead_angle: degrees to turn from current heading
    - intercept_point: [x, y] where intercept occurs
    """
    # Where will target be at intercept time?
    intercept_point = target.position_at_time(intercept_time)

    # Vector from pursuer to intercept point
    to_intercept = intercept_point - pursuer.position

    # Required heading (angle of this vector)
    required_heading = np.degrees(np.arctan2(to_intercept[1], to_intercept[0]))

    # How much does pursuer need to turn?
    lead_angle = required_heading - pursuer.heading

    # Normalize to [-180, 180]
    while lead_angle > 180:
        lead_angle -= 360
    while lead_angle < -180:
        lead_angle += 360

    return lead_angle, intercept_point


def analyze_engagement(pursuer: Aircraft, target: Aircraft) -> dict:
    """
    Complete engagement analysis - can we intercept? How? When?

    Returns a dictionary with all the key information.
    """
    result = {
        "pursuer": str(pursuer),
        "target": str(target),
        "intercept_possible": False,
        "intercept_time": None,
        "intercept_point": None,
        "required_heading": None,
        "lead_angle": None,
        "closure_rate": None,
        "initial_range": None,
    }

    # Initial range (distance between aircraft)
    rel_pos = target.position - pursuer.position
    result["initial_range"] = np.linalg.norm(rel_pos)

    # Closure rate (how fast is the range decreasing?)
    rel_vel = pursuer.velocity - target.velocity
    range_unit = rel_pos / result["initial_range"]
    result["closure_rate"] = np.dot(rel_vel, range_unit)

    # Calculate intercept
    t_intercept = calculate_intercept_time(pursuer, target)

    if t_intercept is not None:
        result["intercept_possible"] = True
        result["intercept_time"] = t_intercept

        lead_angle, intercept_point = calculate_lead_angle(pursuer, target, t_intercept)
        result["lead_angle"] = lead_angle
        result["intercept_point"] = intercept_point
        result["required_heading"] = pursuer.heading + lead_angle

    return result
