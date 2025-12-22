"""
PHASE 1: Intercept Geometry Fundamentals
=========================================
This simulation teaches the core math behind pursuit and intercept problems.

Key concepts:
- Position vectors and relative motion
- Time-to-intercept calculations  
- Lead pursuit (aiming where target WILL BE)
- Engagement envelope (can we even reach them?)

Author: James (learning simulation fundamentals)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto
import time
import random


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
    """Represents an aircraft/missile with position, velocity, and flight dynamics (3D)."""
    name: str
    position: np.ndarray      # [x, y, z] in meters (Earth-centered)
    velocity: np.ndarray      # [vx, vy, vz] in meters/second
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
        """Current heading/azimuth in degrees (0 = +X, 90 = +Y)."""
        return np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))

    @property
    def pitch(self) -> float:
        """Current pitch in degrees (positive = climbing, negative = diving)."""
        horizontal_speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if horizontal_speed < 0.1:
            return 90.0 if self.velocity[2] > 0 else -90.0
        return np.degrees(np.arctan2(self.velocity[2], horizontal_speed))

    def position_at_time(self, t: float) -> np.ndarray:
        """Where will this aircraft be at time t (assuming constant velocity)?"""
        return self.position + self.velocity * t

    def __str__(self):
        burn_status = "BURN" if self.is_burning else "COAST"
        return (f"{self.name}: pos=({self.position[0]/1000:.1f}, {self.position[1]/1000:.1f}, {self.position[2]/1000:.1f})km, "
                f"speed={self.speed:.0f}m/s, hdg={self.heading:.1f}°, pitch={self.pitch:.1f}°, {burn_status}")


@dataclass
class Seeker:
    """
    Models an active radar or IR seeker head (3D).

    The seeker has:
    - Acquisition cone: Target must be within this 3D cone to acquire lock
    - Gimbal limits: Maximum off-boresight angle the seeker can track
    - Lock-on time: Time target must be in cone before lock acquired
    - State machine: OFF -> SEARCHING -> LOCKED (or LOST)
    """

    # Seeker geometry (degrees) - now cone angles for 3D
    acquisition_cone: float = 30.0    # Half-angle for target acquisition (3D cone)
    gimbal_limit: float = 60.0        # Maximum off-boresight tracking angle (3D cone)

    # Seeker performance
    max_range: float = 20_000.0       # Maximum seeker lock range (meters)
    min_range: float = 100.0          # Minimum range (fusing distance)
    lock_time: float = 0.5            # Time to acquire lock (seconds)

    # State tracking
    state: SeekerState = field(default=SeekerState.OFF)
    target_bearing: float = 0.0       # Off-boresight angle to target (degrees, 3D cone angle)
    time_in_cone: float = 0.0         # Time target has been in acquisition cone
    los_rate: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # 3D LOS rate vector (rad/s)
    los_rate_magnitude: float = 0.0   # Magnitude of LOS rate for PN calculations
    closing_velocity: float = 0.0     # Vc = -dR/dt (m/s, positive when closing)

    def update(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
               missile_heading: float, target_pos: np.ndarray,
               target_vel: np.ndarray, dt: float) -> bool:
        """
        Update seeker state based on current 3D geometry.

        Args:
            missile_pos: Missile position [x, y, z]
            missile_vel: Missile velocity [vx, vy, vz]
            missile_heading: Missile heading in degrees (kept for compatibility)
            target_pos: Target position [x, y, z]
            target_vel: Target velocity [vx, vy, vz]
            dt: Time step

        Returns:
            True if target is being tracked (locked)
        """
        # Calculate relative geometry
        to_target = target_pos - missile_pos
        range_to_target = np.linalg.norm(to_target)

        # Calculate 3D off-boresight angle (cone angle from velocity vector to target)
        self._update_bearing_3d(missile_vel, to_target, range_to_target)

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

    def _update_bearing_3d(self, missile_vel: np.ndarray, to_target: np.ndarray,
                           range_to_target: float):
        """Calculate 3D off-boresight angle (angle between velocity and target direction)."""
        if range_to_target < 1.0:
            self.target_bearing = 0.0
            return

        missile_speed = np.linalg.norm(missile_vel)
        if missile_speed < 1.0:
            self.target_bearing = 0.0
            return

        to_target_unit = to_target / range_to_target
        boresight_unit = missile_vel / missile_speed

        # Dot product gives cos(angle)
        cos_angle = np.clip(np.dot(boresight_unit, to_target_unit), -1.0, 1.0)
        self.target_bearing = np.degrees(np.arccos(cos_angle))

    def _update_los_rate(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
                         target_pos: np.ndarray, target_vel: np.ndarray, dt: float):
        """
        Calculate 3D line-of-sight rate vector: omega = (R x V_rel) / |R|^2

        The LOS rate is a 3D angular velocity vector perpendicular to the LOS.
        """
        R = target_pos - missile_pos
        V_rel = target_vel - missile_vel
        R_mag = np.linalg.norm(R)

        if R_mag < 10:
            self.los_rate = np.array([0.0, 0.0, 0.0])
            self.los_rate_magnitude = 0.0
            return

        # 3D LOS rate: omega = (R x V_rel) / |R|^2
        self.los_rate = np.cross(R, V_rel) / (R_mag ** 2)
        self.los_rate_magnitude = np.linalg.norm(self.los_rate)

    def _update_closing_velocity(self, missile_pos: np.ndarray, missile_vel: np.ndarray,
                                  target_pos: np.ndarray, target_vel: np.ndarray):
        """
        Calculate closing velocity: Vc = -dR/dt

        Positive when range is decreasing (closing). Works in any dimension.
        """
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
        """Check if target is within acquisition cone (3D cone angle)."""
        return self.target_bearing <= self.acquisition_cone

    def _is_in_gimbal_limits(self) -> bool:
        """Check if target is within gimbal tracking limits (3D cone angle)."""
        return self.target_bearing <= self.gimbal_limit

    def activate(self):
        """Activate seeker (start searching)."""
        if self.state == SeekerState.OFF:
            self.state = SeekerState.SEARCHING
            self.time_in_cone = 0.0


@dataclass
class TargetEstimator:
    """
    Kalman filter for target state estimation (3D).

    Smooths noisy seeker measurements and predicts target position
    between updates. Uses a constant velocity motion model.

    State vector: [x, y, z, vx, vy, vz] (6 elements)
    Measurements: [x, y, z] (position only from seeker)
    """

    # Process noise (target maneuver capability - higher = more maneuvering expected)
    process_noise: float = 50.0      # m/s² equivalent acceleration noise

    # Measurement noise (seeker accuracy)
    measurement_noise: float = 100.0  # meters position uncertainty

    # Internal state (initialized on first update)
    _initialized: bool = field(default=False, repr=False)
    _x: np.ndarray = field(default=None, repr=False)  # State estimate [x, y, z, vx, vy, vz]
    _P: np.ndarray = field(default=None, repr=False)  # State covariance 6x6

    def predict(self, dt: float):
        """
        Predict step: propagate state forward using constant velocity model (3D).

        State transition: x_new = F @ x
        where F is the 6x6 state transition matrix for constant velocity.
        """
        if not self._initialized:
            return

        # State transition matrix (constant velocity model) - 6x6
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise covariance (discretized continuous white noise) - 6x6
        q = self.process_noise
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = np.array([
            [dt4/4, 0, 0, dt3/2, 0, 0],
            [0, dt4/4, 0, 0, dt3/2, 0],
            [0, 0, dt4/4, 0, 0, dt3/2],
            [dt3/2, 0, 0, dt2, 0, 0],
            [0, dt3/2, 0, 0, dt2, 0],
            [0, 0, dt3/2, 0, 0, dt2]
        ]) * q**2

        # Predict state and covariance
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, measurement: np.ndarray):
        """
        Update step: incorporate position measurement from seeker (3D).

        Args:
            measurement: [x, y, z] position measurement
        """
        if not self._initialized:
            # Initialize state from first measurement
            self._x = np.array([measurement[0], measurement[1], measurement[2],
                               0.0, 0.0, 0.0])
            self._P = np.diag([self.measurement_noise**2] * 3 +
                              [1000.0] * 3)  # High initial velocity uncertainty
            self._initialized = True
            return

        # Measurement matrix (observe position only) - 3x6
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Measurement noise covariance - 3x3
        R = np.eye(3) * self.measurement_noise**2

        # Innovation (measurement residual)
        y = measurement - H @ self._x

        # Innovation covariance
        S = H @ self._P @ H.T + R

        # Kalman gain
        K = self._P @ H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(6)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T

    def reset(self):
        """Reset the estimator (for new engagement)."""
        self._initialized = False
        self._x = None
        self._P = None

    @property
    def position(self) -> Optional[np.ndarray]:
        """Estimated target position [x, y, z]."""
        if not self._initialized:
            return None
        return self._x[:3].copy()

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Estimated target velocity [vx, vy, vz]."""
        if not self._initialized:
            return None
        return self._x[3:].copy()

    @property
    def state(self) -> Optional[np.ndarray]:
        """Full state estimate [x, y, z, vx, vy, vz]."""
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

        Returns:
            Current guidance phase
        """
        old_phase = self.current_phase

        # BOOST -> MIDCOURSE: After boost duration
        if (self.current_phase == GuidancePhase.BOOST and
                time_since_launch >= self.boost_duration):
            self.current_phase = GuidancePhase.MIDCOURSE
            self.phase_start_time = time_since_launch

        # MIDCOURSE -> TERMINAL: When seeker locks AND target is reasonably centered
        if (self.current_phase == GuidancePhase.MIDCOURSE and
                seeker.state == SeekerState.LOCKED and
                abs(seeker.target_bearing) < 40.0):  # Within 40 degrees of boresight
            self.current_phase = GuidancePhase.TERMINAL
            self.phase_start_time = time_since_launch

        # TERMINAL: Stay in terminal once entered - don't fall back
        # The terminal guidance with pursuit blending will help re-acquire
        # Falling back to midcourse causes oscillation that hurts intercepts

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

def calculate_intercept_time(pursuer: Aircraft, target: Aircraft, use_current_speed: bool = False) -> Optional[float]:
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

    Args:
        use_current_speed: If True, use current speed instead of max_speed.
                          Better for accelerating missiles.

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
    # |v_t|² - v_p² (if target is faster, this is positive and intercept may be impossible)
    a = np.dot(v_t, v_t) - v_p**2
    
    # 2 * (rel_pos · v_t)
    b = 2 * np.dot(rel_pos, v_t)
    
    # |rel_pos|²
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
    
    THE MATH:
    ---------
    Lead pursuit means aiming at where the target WILL BE, not where it IS.
    
    intercept_point = target_position + target_velocity * intercept_time
    required_heading = direction from pursuer to intercept_point
    
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
    # Positive = getting closer, Negative = getting farther
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


# =============================================================================
# REAL-TIME SIMULATION ENGINE
# =============================================================================

@dataclass
class SimulationState:
    """Tracks the state of a running simulation."""
    time: float = 0.0
    pursuer_path: List[np.ndarray] = field(default_factory=list)
    target_path: List[np.ndarray] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)
    range_history: List[float] = field(default_factory=list)
    status: str = "running"  # running, intercepted, escaped, timeout
    intercept_range: float = 50.0  # meters - considered "hit" if within this range

    # Radar tracking
    radar_status: str = "searching"  # searching, detected, launched
    detection_time: Optional[float] = None  # When target was first detected
    launch_time: Optional[float] = None  # When SAM was launched
    radar_status_history: List[str] = field(default_factory=list)  # Track status over time

    # Flight dynamics tracking
    pursuer_speed_history: List[float] = field(default_factory=list)
    target_speed_history: List[float] = field(default_factory=list)
    pursuer_altitude_history: List[float] = field(default_factory=list)
    target_altitude_history: List[float] = field(default_factory=list)
    pursuer_burn_history: List[bool] = field(default_factory=list)  # True = burning, False = coasting
    pursuer_g_history: List[float] = field(default_factory=list)  # G-forces experienced

    # Advanced guidance tracking
    guidance_phase_history: List[GuidancePhase] = field(default_factory=list)
    seeker_state_history: List[SeekerState] = field(default_factory=list)
    los_rate_history: List[float] = field(default_factory=list)  # Line-of-sight rate (rad/s)
    closing_velocity_history: List[float] = field(default_factory=list)  # Vc (m/s)
    target_bearing_history: List[float] = field(default_factory=list)  # Seeker bearing (deg)


class InterceptSimulation:
    """
    Real-time simulation of pursuit and intercept.

    This class runs a step-by-step simulation where:
    - Target maneuvers randomly
    - Pursuer continuously recalculates intercept and adjusts heading
    - Positions update each timestep
    - Simulation ends on intercept, escape, or timeout
    """

    def __init__(self, pursuer: Aircraft, target: Aircraft,
                 dt: float = 0.1, max_time: float = 120.0,
                 intercept_range: float = 50.0,
                 pursuer_turn_rate: float = 30.0,
                 target_turn_rate: float = 15.0,
                 target_maneuver_intensity: float = 0.3,
                 evasion_radius: float = 3000.0,
                 radar_range: float = 400_000.0,
                 launch_delay: float = 5.0,
                 earth_radius: float = 6_371_000.0,
                 enable_physics: bool = True):
        """
        Initialize simulation.

        Args:
            pursuer: The pursuing aircraft
            target: The target aircraft
            dt: Time step in seconds
            max_time: Maximum simulation time before timeout
            intercept_range: Distance at which intercept is considered successful
            pursuer_turn_rate: Max turn rate for pursuer in degrees/second (base rate, modified by g-limits)
            target_turn_rate: Max turn rate for target in degrees/second
            target_maneuver_intensity: How often/aggressively target turns (0-1)
            evasion_radius: Distance at which target starts actively evading
            radar_range: Maximum radar detection range in meters
            launch_delay: Seconds between detection and launch
            earth_radius: Earth radius for radar horizon calculation
            enable_physics: Enable realistic flight dynamics (gravity, drag, g-limits)
        """
        # Store initial states (deep copy positions/velocities and flight dynamics)
        self.pursuer = Aircraft(
            name=pursuer.name,
            position=pursuer.position.copy(),
            velocity=pursuer.velocity.copy(),
            max_speed=pursuer.max_speed,
            max_acceleration=pursuer.max_acceleration,
            max_g=pursuer.max_g,
            burn_time=pursuer.burn_time,
            drag_coefficient=pursuer.drag_coefficient,
            is_missile=pursuer.is_missile
        )
        self.target = Aircraft(
            name=target.name,
            position=target.position.copy(),
            velocity=target.velocity.copy(),
            max_speed=target.max_speed,
            max_acceleration=target.max_acceleration,
            max_g=target.max_g,
            burn_time=target.burn_time,
            drag_coefficient=target.drag_coefficient,
            is_missile=target.is_missile
        )

        self.dt = dt
        self.max_time = max_time
        self.intercept_range = intercept_range
        self.pursuer_turn_rate = pursuer_turn_rate
        self.target_turn_rate = target_turn_rate
        self.target_maneuver_intensity = target_maneuver_intensity
        self.evasion_radius = evasion_radius

        # Radar system parameters
        self.radar_range = radar_range
        self.launch_delay = launch_delay
        self.earth_radius = earth_radius

        # Physics simulation
        self.enable_physics = enable_physics
        self.GRAVITY = 9.81  # m/s² - gravitational acceleration

        # Store initial SAM position (before launch, SAM is stationary)
        self.sam_launch_position = pursuer.position.copy()

        # Store initial target heading direction (for returning to course after evasion)
        self.target_initial_velocity = target.velocity.copy()
        self.city_position = None  # Set externally for target re-acquisition

        # Advanced guidance system
        self.seeker = Seeker(
            acquisition_cone=30.0,   # ±30° acquisition cone
            gimbal_limit=45.0,       # ±45° gimbal limits
            max_range=20_000.0,      # 20km max seeker range
            lock_time=0.5            # 0.5s to acquire lock
        )
        self.guidance = GuidanceController(
            boost_duration=3.0,      # 3s boost phase
            terminal_range=15_000.0, # Terminal guidance below 15km
            N_midcourse=3.0,         # Coarse PN for midcourse
            N_terminal=4.0           # Fine PN for terminal
        )
        self.target_estimator = TargetEstimator(
            process_noise=50.0,      # Target maneuver capability
            measurement_noise=100.0  # Seeker accuracy
        )
        self.enable_advanced_guidance = True  # Use new multi-phase guidance

        # Initialize state
        self.state = SimulationState(intercept_range=intercept_range)
        self._record_state()

    def _record_state(self, current_g: float = 0.0):
        """Record current positions and flight dynamics for history."""
        self.state.pursuer_path.append(self.pursuer.position.copy())
        self.state.target_path.append(self.target.position.copy())
        self.state.time_history.append(self.state.time)

        current_range = np.linalg.norm(self.target.position - self.pursuer.position)
        self.state.range_history.append(current_range)

        # Record radar status
        self.state.radar_status_history.append(self.state.radar_status)

        # Record flight dynamics
        self.state.pursuer_speed_history.append(self.pursuer.speed)
        self.state.target_speed_history.append(self.target.speed)

        # Calculate altitudes (distance from Earth center minus Earth radius)
        pursuer_alt = np.linalg.norm(self.pursuer.position) - self.earth_radius
        target_alt = np.linalg.norm(self.target.position) - self.earth_radius
        self.state.pursuer_altitude_history.append(pursuer_alt)
        self.state.target_altitude_history.append(target_alt)

        # Record burn status and g-forces
        self.state.pursuer_burn_history.append(self.pursuer.is_burning)
        self.state.pursuer_g_history.append(current_g)

        # Record advanced guidance state
        self.state.guidance_phase_history.append(self.guidance.current_phase)
        self.state.seeker_state_history.append(self.seeker.state)
        self.state.los_rate_history.append(self.seeker.los_rate)
        self.state.closing_velocity_history.append(self.seeker.closing_velocity)
        self.state.target_bearing_history.append(self.seeker.target_bearing)

    def _calculate_radar_horizon(self) -> float:
        """
        Calculate radar horizon distance based on Earth's curvature.

        The radar horizon is the maximum distance at which a target at a given
        altitude can be detected, limited by Earth's curvature.

        Formula: d = sqrt(2 * R * h_radar) + sqrt(2 * R * h_target)
        where R is Earth radius, h is altitude above surface.
        """
        # SAM radar altitude (distance from Earth center minus Earth radius)
        sam_altitude = np.linalg.norm(self.sam_launch_position) - self.earth_radius
        sam_altitude = max(0, sam_altitude)  # Ensure non-negative

        # Target altitude
        target_altitude = np.linalg.norm(self.target.position) - self.earth_radius
        target_altitude = max(0, target_altitude)

        # Radar horizon formula (geometric horizon + target visibility)
        radar_horizon = (np.sqrt(2 * self.earth_radius * sam_altitude) +
                        np.sqrt(2 * self.earth_radius * target_altitude))

        return radar_horizon

    def _check_radar_detection(self) -> bool:
        """
        Check if target is detected by radar.

        Detection requires:
        1. Target within radar range
        2. Target above radar horizon (line of sight not blocked by Earth)
        """
        # Calculate range to target from SAM site (not current pursuer position)
        range_to_target = np.linalg.norm(self.target.position - self.sam_launch_position)

        # Calculate radar horizon
        radar_horizon = self._calculate_radar_horizon()

        # Target must be within radar range AND above horizon
        within_range = range_to_target <= self.radar_range
        above_horizon = range_to_target <= radar_horizon

        return within_range and above_horizon

    def _get_gravity_vector(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate gravity vector pointing toward Earth's center.

        In our Earth-centered coordinate system, gravity always points
        toward the origin (0, 0) which is Earth's center.
        """
        distance_from_center = np.linalg.norm(position)
        if distance_from_center < 1.0:  # Avoid division by zero
            return np.array([0.0, 0.0])

        # Unit vector pointing toward Earth's center (negative of position direction)
        gravity_direction = -position / distance_from_center
        return self.GRAVITY * gravity_direction

    def _calculate_drag(self, aircraft: Aircraft) -> np.ndarray:
        """
        Calculate drag force (opposes velocity).

        Simplified drag model: F_drag = -k * v² * v_hat
        where k is drag coefficient and v is speed.
        """
        speed = aircraft.speed
        if speed < 1.0:  # Avoid issues at very low speeds
            return np.array([0.0, 0.0])

        # Drag opposes velocity direction, proportional to v²
        velocity_unit = aircraft.velocity / speed
        drag_magnitude = aircraft.drag_coefficient * speed * speed

        return -drag_magnitude * velocity_unit

    def _calculate_max_turn_rate(self, aircraft: Aircraft) -> float:
        """
        Calculate maximum turn rate based on g-limit and speed.

        Turn rate is limited by: ω = g_max * g / v
        where g_max is max sustainable g-load, g is gravity, v is speed.

        At higher speeds, you need more g-force to turn the same amount,
        so turn rate decreases.
        """
        speed = aircraft.speed
        if speed < 10.0:  # Very low speed, use base turn rate
            return self.pursuer_turn_rate

        # Maximum centripetal acceleration = max_g * gravity
        max_centripetal = aircraft.max_g * self.GRAVITY

        # Turn rate (rad/s) = centripetal_accel / speed
        max_turn_rate_rad = max_centripetal / speed
        max_turn_rate_deg = np.degrees(max_turn_rate_rad)

        # Return the minimum of g-limited rate and base rate
        return min(max_turn_rate_deg, self.pursuer_turn_rate)

    def _apply_physics(self, aircraft: Aircraft, thrust_direction: np.ndarray = None) -> float:
        """
        Apply physics to an aircraft/missile for one timestep.

        Applies:
        1. Gravity (for missiles)
        2. Drag (speed-dependent)
        3. Thrust (if burning and direction provided)

        Returns:
            Current g-force experienced (for tracking)
        """
        if not self.enable_physics:
            return 0.0

        # Start with zero acceleration
        acceleration = np.array([0.0, 0.0])

        # 1. Gravity (only for missiles, not aircraft maintaining altitude)
        if aircraft.is_missile:
            gravity = self._get_gravity_vector(aircraft.position)
            acceleration += gravity

        # 2. Drag (always applies)
        drag_accel = self._calculate_drag(aircraft)
        acceleration += drag_accel

        # 3. Thrust (only if burning and direction provided)
        if aircraft.is_burning and thrust_direction is not None:
            thrust_mag = np.linalg.norm(thrust_direction)
            if thrust_mag > 0:
                thrust_unit = thrust_direction / thrust_mag
                acceleration += aircraft.max_acceleration * thrust_unit

        # Apply acceleration to velocity
        aircraft.velocity += acceleration * self.dt

        # Calculate g-force experienced (total acceleration / gravity)
        g_force = np.linalg.norm(acceleration) / self.GRAVITY

        return g_force

    def _update_burn_status(self, aircraft: Aircraft):
        """
        Update fuel/burn status based on time since launch.

        After burn_time seconds, missile switches to coast phase
        (no more thrust, just momentum + gravity + drag).
        """
        if aircraft.time_since_launch >= aircraft.burn_time:
            if aircraft.is_burning:
                aircraft.is_burning = False
                print(f"  >>> {aircraft.name} BURNOUT at t={self.state.time:.1f}s - now coasting!")

    # =========================================================================
    # ADVANCED GUIDANCE METHODS
    # =========================================================================

    def _calculate_pn_acceleration(self, N: float = 4.0) -> np.ndarray:
        """
        Calculate commanded acceleration using true Proportional Navigation.

        PN Law: a = N × Vc × (dλ/dt) × n̂

        Where:
        - N: Navigation constant (3-5 typical, 4 is common for terminal)
        - Vc: Closing velocity (positive when closing)
        - dλ/dt: Line-of-sight rate (rad/s)
        - n̂: Unit vector perpendicular to line-of-sight (turn plane)

        Args:
            N: Navigation constant

        Returns:
            Commanded acceleration vector [ax, ay] in m/s²
        """
        # Get seeker measurements
        los_rate = self.seeker.los_rate
        Vc = self.seeker.closing_velocity

        # Safety check
        if Vc < 10:  # Not closing meaningfully
            return np.array([0.0, 0.0])

        # Calculate LOS unit vector (from missile to target)
        to_target = self.target.position - self.pursuer.position
        range_to_target = np.linalg.norm(to_target)

        if range_to_target < 10:
            return np.array([0.0, 0.0])

        los_unit = to_target / range_to_target

        # Calculate perpendicular vector (n̂) - direction of turn
        # For 2D: rotate LOS by 90 degrees
        n_hat = np.array([-los_unit[1], los_unit[0]])

        # PN commanded acceleration magnitude
        # Note: los_rate has sign indicating turn direction
        a_magnitude = N * Vc * abs(los_rate)

        # Apply in correct direction based on LOS rate sign
        a_commanded = a_magnitude * n_hat * np.sign(los_rate)

        return a_commanded

    def _boost_phase_guidance(self) -> np.ndarray:
        """
        Boost phase: Fixed attitude toward predicted intercept point.

        During boost, the missile climbs rapidly without active guidance.
        Just maintain initial aim toward predicted intercept.

        Returns:
            Thrust direction unit vector
        """
        # Calculate predicted intercept at current time
        t_intercept = calculate_intercept_time(self.pursuer, self.target)

        if t_intercept is not None and t_intercept > 0:
            intercept_point = self.target.position_at_time(t_intercept)
            direction = intercept_point - self.pursuer.position
        else:
            direction = self.target.position - self.pursuer.position

        direction = direction / np.linalg.norm(direction)

        # Add initial climb bias (typical for SAMs) - reduced since gravity
        # compensation is also applied separately after turn rate limiting
        up_direction = self.pursuer.position / np.linalg.norm(self.pursuer.position)
        thrust_direction = 0.85 * direction + 0.15 * up_direction
        thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)

        return thrust_direction

    def _midcourse_guidance(self) -> np.ndarray:
        """
        Midcourse phase: Command guidance from ground radar.

        Ground radar provides periodic updates. Missile uses lead pursuit
        with coarse PN corrections. Seeker is searching for lock.

        Returns:
            Thrust direction unit vector
        """
        current_time = self.state.time

        # Get ground radar update (periodic)
        if self.guidance.should_update_radar(current_time):
            # Ground radar calculates intercept point
            t_intercept = calculate_intercept_time(self.pursuer, self.target)
            if t_intercept is not None and t_intercept > 0:
                self.guidance.midcourse_aim_point = self.target.position_at_time(t_intercept)
            else:
                self.guidance.midcourse_aim_point = self.target.position.copy()

            # Update Kalman filter with ground radar measurement
            self.target_estimator.predict(1.0 / self.guidance.ground_radar_update_rate)
            self.target_estimator.update(self.target.position)

        # Aim toward stored aim point
        if self.guidance.midcourse_aim_point is not None:
            direction = self.guidance.midcourse_aim_point - self.pursuer.position
        else:
            direction = self.target.position - self.pursuer.position

        range_to_aim = np.linalg.norm(direction)
        if range_to_aim > 10:
            direction = direction / range_to_aim

            # Add coarse PN correction if seeker has data
            if self.seeker.state in [SeekerState.SEARCHING, SeekerState.LOCKED]:
                pn_accel = self._calculate_pn_acceleration(N=self.guidance.N_midcourse)
                accel_mag = np.linalg.norm(pn_accel)

                # Blend PN with lead pursuit (PN weight increases as we get closer)
                if accel_mag > 0.1:
                    pn_weight = min(0.3, accel_mag / (self.pursuer.max_g * self.GRAVITY))
                    direction = (1 - pn_weight) * direction + pn_weight * (pn_accel / accel_mag)
                    direction = direction / np.linalg.norm(direction)

        # Activate seeker if not already
        if self.seeker.state == SeekerState.OFF:
            self.seeker.activate()

        return direction

    def _terminal_guidance(self) -> np.ndarray:
        """
        Terminal phase: Active seeker with true Proportional Navigation.

        Seeker is locked on target. Use pure PN law for precise terminal
        guidance with Kalman-filtered target state.

        Returns:
            Thrust direction unit vector
        """
        # Update Kalman filter with seeker measurements
        self.target_estimator.predict(self.dt)
        self.target_estimator.update(self.target.position)

        # Get estimated target state
        est_pos = self.target_estimator.position
        if est_pos is None:
            est_pos = self.target.position

        # Pure PN law
        pn_accel = self._calculate_pn_acceleration(N=self.guidance.N_terminal)

        # Current velocity direction
        if self.pursuer.speed > 10:
            current_direction = self.pursuer.velocity / self.pursuer.speed
        else:
            current_direction = np.array([1.0, 0.0])

        # Apply PN correction
        accel_mag = np.linalg.norm(pn_accel)
        if accel_mag > 0.1:
            # Weight PN correction by available g
            max_lateral_accel = self.pursuer.max_g * self.GRAVITY
            correction_weight = min(0.8, accel_mag / max_lateral_accel)

            accel_unit = pn_accel / accel_mag
            thrust_direction = (1 - correction_weight) * current_direction + correction_weight * accel_unit
            thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
        else:
            thrust_direction = current_direction

        return thrust_direction

    def _apply_gravity_compensation(self, thrust_direction: np.ndarray) -> np.ndarray:
        """
        Apply gravity compensation to thrust direction.

        Adds upward component to counter gravitational pull.

        Args:
            thrust_direction: Desired thrust direction

        Returns:
            Gravity-compensated thrust direction
        """
        if not (self.pursuer.is_missile and self.enable_physics and self.pursuer.is_burning):
            return thrust_direction

        gravity = self._get_gravity_vector(self.pursuer.position)
        up_direction = -gravity / np.linalg.norm(gravity)

        to_target = self.target.position - self.pursuer.position
        range_to_target = np.linalg.norm(to_target)

        if range_to_target > 100:
            vertical_component = np.dot(to_target, up_direction)
            climb_ratio = vertical_component / range_to_target
            gravity_counter_weight = max(0, climb_ratio) * 0.08 + 0.05

            thrust_direction = thrust_direction + up_direction * gravity_counter_weight
            thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)

        return thrust_direction

    def _update_pursuer_heading(self) -> np.ndarray:
        """
        Recalculate intercept and adjust pursuer heading.

        With advanced guidance enabled:
        - Updates seeker state (tracking, LOS rate, closing velocity)
        - Updates guidance phase (BOOST -> MIDCOURSE -> TERMINAL)
        - Uses phase-appropriate guidance law
        - Applies gravity compensation

        Without advanced guidance (legacy mode):
        - Uses lead pursuit with gravity compensation

        Returns:
            Desired thrust direction (unit vector) for physics application
        """
        # =====================================================================
        # ADVANCED MULTI-PHASE GUIDANCE
        # =====================================================================
        if self.enable_advanced_guidance:
            # Update seeker with current geometry
            self.seeker.update(
                self.pursuer.position,
                self.pursuer.velocity,
                self.pursuer.heading,
                self.target.position,
                self.target.velocity,
                self.dt
            )

            # Update guidance phase
            range_to_target = np.linalg.norm(self.target.position - self.pursuer.position)
            phase = self.guidance.update_phase(
                self.pursuer.time_since_launch,
                range_to_target,
                self.seeker
            )

            # Get thrust direction from phase-specific guidance
            if phase == GuidancePhase.BOOST:
                thrust_direction = self._boost_phase_guidance()
            elif phase == GuidancePhase.MIDCOURSE:
                thrust_direction = self._midcourse_guidance()
            elif phase == GuidancePhase.TERMINAL:
                thrust_direction = self._terminal_guidance()
            else:
                # Fallback to lead pursuit
                thrust_direction = self._legacy_lead_pursuit()

            # Apply turn rate limiting FIRST (to get heading-limited direction)
            thrust_direction = self._apply_turn_rate_limit(thrust_direction)

            # Apply gravity compensation AFTER turn limiting
            # (This ensures the compensation isn't lost during heading conversion)
            thrust_direction = self._apply_gravity_compensation(thrust_direction)

            if not self.enable_physics:
                self.pursuer.velocity = self.pursuer.max_speed * thrust_direction

            return thrust_direction

        # =====================================================================
        # LEGACY LEAD PURSUIT (backward compatibility)
        # =====================================================================
        return self._legacy_lead_pursuit()

    def _legacy_lead_pursuit(self) -> np.ndarray:
        """
        Original lead pursuit guidance with gravity compensation.

        Used when advanced guidance is disabled.
        """
        t_intercept = calculate_intercept_time(self.pursuer, self.target, use_current_speed=False)

        if t_intercept is not None and t_intercept > 0:
            intercept_point = self.target.position_at_time(t_intercept)
            desired_direction = intercept_point - self.pursuer.position
        else:
            desired_direction = self.target.position - self.pursuer.position

        desired_heading_rad = np.arctan2(desired_direction[1], desired_direction[0])
        desired_heading = np.degrees(desired_heading_rad)

        heading_diff = desired_heading - self.pursuer.heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        if self.enable_physics:
            max_turn_rate = self._calculate_max_turn_rate(self.pursuer)
        else:
            max_turn_rate = self.pursuer_turn_rate

        max_turn = max_turn_rate * self.dt
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)

        new_heading = self.pursuer.heading + actual_turn
        new_heading_rad = np.radians(new_heading)

        velocity_direction = np.array([np.cos(new_heading_rad), np.sin(new_heading_rad)])

        # Apply gravity compensation
        thrust_direction = self._apply_gravity_compensation(velocity_direction)

        if not self.enable_physics:
            self.pursuer.velocity = self.pursuer.max_speed * thrust_direction

        return thrust_direction

    def _apply_turn_rate_limit(self, desired_direction: np.ndarray) -> np.ndarray:
        """
        Limit how fast the missile can change direction.

        Args:
            desired_direction: Desired thrust direction

        Returns:
            Turn-rate-limited thrust direction
        """
        # Current heading
        current_heading = self.pursuer.heading

        # Desired heading
        desired_heading = np.degrees(np.arctan2(desired_direction[1], desired_direction[0]))

        # Heading difference
        heading_diff = desired_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # Get max turn rate
        if self.enable_physics:
            max_turn_rate = self._calculate_max_turn_rate(self.pursuer)
        else:
            max_turn_rate = self.pursuer_turn_rate

        max_turn = max_turn_rate * self.dt

        # Limit turn
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)
        new_heading = current_heading + actual_turn
        new_heading_rad = np.radians(new_heading)

        return np.array([np.cos(new_heading_rad), np.sin(new_heading_rad)])

    def _update_target_heading(self):
        """
        Update target heading - evade if pursuer is close, otherwise head toward city.

        When pursuer is within evasion_radius, target actively steers away
        and applies a temporary speed boost.
        Otherwise, steers back toward the city/original course.
        """
        # Calculate distance to pursuer
        to_pursuer = self.pursuer.position - self.target.position
        distance_to_pursuer = np.linalg.norm(to_pursuer)

        max_turn = self.target_turn_rate * self.dt
        is_evading = False

        if distance_to_pursuer <= self.evasion_radius:
            # EVASION MODE: Steer away from pursuer
            away_from_pursuer = -to_pursuer
            desired_heading = np.degrees(np.arctan2(away_from_pursuer[1], away_from_pursuer[0]))
            is_evading = True

        elif self.city_position is not None:
            # RE-ACQUISITION MODE: Steer toward city
            to_city = self.city_position - self.target.position
            desired_heading = np.degrees(np.arctan2(to_city[1], to_city[0]))

        else:
            # Fallback: return to original course direction
            desired_heading = np.degrees(np.arctan2(
                self.target_initial_velocity[1],
                self.target_initial_velocity[0]
            ))

        # Calculate turn needed
        heading_diff = desired_heading - self.target.heading

        # Normalize to [-180, 180]
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # Only turn if we need to (small deadband to avoid jitter)
        if abs(heading_diff) < 0.5 and not is_evading:
            return

        # Apply turn rate limit
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)
        new_heading = self.target.heading + actual_turn

        # Maintain cruise speed (no boost during evasion - turning bleeds energy)
        target_speed = self.target.speed

        # Apply new heading and speed
        new_heading_rad = np.radians(new_heading)
        self.target.velocity = target_speed * np.array([
            np.cos(new_heading_rad),
            np.sin(new_heading_rad)
        ])

    def _check_termination(self) -> bool:
        """Check if simulation should end."""
        current_range = np.linalg.norm(self.target.position - self.pursuer.position)

        # Check intercept
        if current_range <= self.intercept_range:
            self.state.status = "intercepted"
            return True

        # Check timeout
        if self.state.time >= self.max_time:
            self.state.status = "timeout"
            return True

        # Check if target is escaping (range increasing and no intercept possible)
        if len(self.state.range_history) > 10:
            recent_ranges = self.state.range_history[-10:]
            if all(recent_ranges[i] < recent_ranges[i+1] for i in range(len(recent_ranges)-1)):
                # Range has been monotonically increasing
                if calculate_intercept_time(self.pursuer, self.target) is None:
                    self.state.status = "escaped"
                    return True

        return False

    def _update_radar_status(self):
        """
        Update radar detection and launch status.

        Phases:
        1. SEARCHING: Radar scanning, SAM stationary
        2. DETECTED: Target acquired, preparing to launch (launch_delay countdown)
        3. LAUNCHED: SAM is in flight, pursuing target
        """
        if self.state.radar_status == "searching":
            # Check for target detection
            if self._check_radar_detection():
                self.state.radar_status = "detected"
                self.state.detection_time = self.state.time
                print(f"  >>> RADAR CONTACT at t={self.state.time:.1f}s!")

        elif self.state.radar_status == "detected":
            # Check if launch delay has elapsed
            time_since_detection = self.state.time - self.state.detection_time
            if time_since_detection >= self.launch_delay:
                self.state.radar_status = "launched"
                self.state.launch_time = self.state.time

                # Set initial launch velocity toward target (or intercept point)
                t_intercept = calculate_intercept_time(self.pursuer, self.target)
                if t_intercept is not None:
                    intercept_point = self.target.position_at_time(t_intercept)
                    launch_dir = intercept_point - self.pursuer.position
                else:
                    launch_dir = self.target.position - self.pursuer.position

                launch_dir_unit = launch_dir / np.linalg.norm(launch_dir)
                initial_speed = 500.0  # Initial boost velocity
                self.pursuer.velocity = initial_speed * launch_dir_unit

                print(f"  >>> SAM LAUNCHED at t={self.state.time:.1f}s!")

    def step(self) -> bool:
        """
        Execute one simulation step.

        Returns:
            True if simulation should continue, False if terminated.
        """
        if self.state.status != "running":
            return False

        current_g = 0.0  # Track g-forces for recording

        # Update target maneuvers (random turns)
        self._update_target_heading()

        # Update radar detection status
        self._update_radar_status()

        # Only update pursuer if SAM has been launched
        if self.state.radar_status == "launched":
            # Track time since launch for fuel calculations
            self.pursuer.time_since_launch += self.dt

            # Check and update burn status (fuel depletion)
            self._update_burn_status(self.pursuer)

            # Update pursuer guidance - get thrust direction
            thrust_direction = self._update_pursuer_heading()

            if self.enable_physics:
                # Apply physics: gravity, drag, thrust
                current_g = self._apply_physics(self.pursuer, thrust_direction)

            # Update pursuer position
            self.pursuer.position += self.pursuer.velocity * self.dt

            # Check if missile crashed into Earth
            pursuer_alt = np.linalg.norm(self.pursuer.position) - self.earth_radius
            if pursuer_alt < 0:
                print(f"  >>> SAM CRASHED at t={self.state.time:.1f}s!")
                self.state.status = "escaped"  # SAM failed, target escapes
                self._record_state(current_g)
                return False
        else:
            # SAM is stationary at launch site - keep it there
            self.pursuer.position = self.sam_launch_position.copy()
            self.pursuer.velocity = np.array([0.0, 0.0])

        # Apply physics to target (gravity, drag - no thrust since cruise missile coasts)
        if self.enable_physics:
            # Target cruise missile - apply gravity and drag
            self._apply_physics(self.target, thrust_direction=None)

        # Target always moves
        self.target.position += self.target.velocity * self.dt

        # Check if target hit Earth (or city!)
        target_alt = np.linalg.norm(self.target.position) - self.earth_radius
        if target_alt < 0:
            print(f"  >>> TARGET IMPACTED at t={self.state.time:.1f}s!")
            self.state.status = "escaped"  # Target reached ground
            self._record_state(current_g)
            return False

        # Advance time
        self.state.time += self.dt

        # Record state with current g-force
        self._record_state(current_g)

        # Check termination
        if self._check_termination():
            return False

        return True

    def run(self, verbose: bool = True) -> SimulationState:
        """
        Run the complete simulation.

        Args:
            verbose: Print progress updates

        Returns:
            Final simulation state
        """
        if verbose:
            print(f"\n--- SIMULATION START ---")
            print(f"Pursuer: {self.pursuer}")
            print(f"Target: {self.target}")
            print(f"Time step: {self.dt}s, Max time: {self.max_time}s")
            print(f"Intercept range: {self.intercept_range}m")
            print(f"Radar range: {self.radar_range/1000:.0f}km, Launch delay: {self.launch_delay}s")
            print(f"Radar horizon: {self._calculate_radar_horizon()/1000:.0f}km")
            if self.enable_physics:
                print(f"\n--- FLIGHT DYNAMICS ENABLED ---")
                print(f"SAM: accel={self.pursuer.max_acceleration:.0f}m/s², max-g={self.pursuer.max_g:.0f}g, burn={self.pursuer.burn_time:.0f}s")
                print(f"Target: drag={self.target.drag_coefficient:.4f}")

        step_count = 0
        while self.step():
            step_count += 1

            # Progress update every 5 seconds of sim time
            if verbose and step_count % int(5.0 / self.dt) == 0:
                current_range = self.state.range_history[-1]
                print(f"  t={self.state.time:.1f}s: range={current_range:.0f}m, "
                      f"pursuer heading={self.pursuer.heading:.1f}°")

        if verbose:
            print(f"\n--- SIMULATION END ---")
            print(f"Status: {self.state.status.upper()}")
            print(f"Final time: {self.state.time:.2f}s")
            print(f"Final range: {self.state.range_history[-1]:.1f}m")
            if self.state.detection_time is not None:
                print(f"Detection time: {self.state.detection_time:.1f}s")
            if self.state.launch_time is not None:
                print(f"Launch time: {self.state.launch_time:.1f}s")
                flight_time = self.state.time - self.state.launch_time
                print(f"SAM flight time: {flight_time:.1f}s")
            print(f"Total steps: {len(self.state.time_history)}")

        return self.state

    def run_realtime(self, speed_multiplier: float = 10.0,
                     callback=None) -> SimulationState:
        """
        Run simulation in real-time (or accelerated).

        Args:
            speed_multiplier: How much faster than real-time to run
            callback: Optional function called each step with (sim, state)

        Returns:
            Final simulation state
        """
        print(f"\n--- REAL-TIME SIMULATION ({speed_multiplier}x speed) ---")
        print("Press Ctrl+C to stop early\n")

        real_dt = self.dt / speed_multiplier

        try:
            while self.step():
                if callback:
                    callback(self, self.state)

                # Print status
                current_range = self.state.range_history[-1]
                print(f"\rt={self.state.time:6.1f}s | range={current_range:8.0f}m | "
                      f"heading={self.pursuer.heading:6.1f}° | status={self.state.status}",
                      end="", flush=True)

                time.sleep(real_dt)

        except KeyboardInterrupt:
            self.state.status = "interrupted"
            print("\n\nSimulation interrupted by user.")

        print(f"\n\n--- FINAL: {self.state.status.upper()} at t={self.state.time:.2f}s ---")
        return self.state


# =============================================================================
# SATURATION ATTACK SIMULATION
# =============================================================================

@dataclass
class SaturationState:
    """State tracking for saturation attack simulation."""
    time: float = 0.0
    status: str = "running"  # running, completed

    # Target tracking
    targets_total: int = 0
    targets_destroyed: int = 0
    targets_leaked: int = 0
    targets_incoming: int = 0

    # Interceptor tracking
    interceptors_total: int = 0
    interceptors_launched: int = 0
    interceptors_remaining: int = 0
    interceptors_in_flight: int = 0

    # Event log for animation
    events: List[Tuple[float, str, int, Optional[int]]] = field(default_factory=list)
    # (time, event_type, target_id, interceptor_id)
    # event_type: "launch", "hit", "miss", "leak"


class SaturationSimulation:
    """
    Simulates a saturation attack scenario with multiple incoming cruise missiles
    and a SAM battery defending a city.

    Features:
    - Multiple incoming targets with staggered arrival
    - SAM battery with limited inventory and reload delays
    - Threat prioritization based on time-to-impact
    - Realistic probability of kill (Pk) modeling
    - Multi-target tracking and engagement
    """

    def __init__(self,
                 num_targets: int = 6,
                 num_interceptors: int = 8,
                 pk_single: float = 0.85,
                 reload_time: float = 8.0,
                 max_simultaneous: int = 4,
                 city_position: np.ndarray = None,
                 battery_position: np.ndarray = None,
                 earth_radius: float = 6_371_000.0,
                 dt: float = 0.1,
                 max_time: float = 300.0,
                 intercept_range: float = 200.0):
        """
        Initialize saturation attack simulation.

        Args:
            num_targets: Number of incoming cruise missiles
            num_interceptors: SAM battery inventory
            pk_single: Probability of kill per intercept attempt
            reload_time: Seconds between SAM launches
            max_simultaneous: Max interceptors in flight at once
            city_position: [x, y] position of city being defended
            battery_position: [x, y] position of SAM battery
            earth_radius: Earth radius for physics calculations
            dt: Simulation time step
            max_time: Maximum simulation duration
            intercept_range: Distance for successful intercept (meters)
        """
        self.num_targets = num_targets
        self.num_interceptors = num_interceptors
        self.dt = dt
        self.max_time = max_time
        self.intercept_range = intercept_range
        self.earth_radius = earth_radius
        self.GRAVITY = 9.81

        # Set default 3D positions if not provided
        if city_position is None:
            # City at "top" of Earth sphere (0, 0, earth_radius)
            city_position = np.array([0.0, 0.0, earth_radius])
        if battery_position is None:
            # SAM site 50km from city in -X direction (between city and incoming threats)
            sam_angle = 50_000 / earth_radius
            battery_position = np.array([
                -earth_radius * np.sin(sam_angle),  # Negative X (toward threats)
                0.0,
                earth_radius * np.cos(sam_angle)
            ])

        self.city_position = city_position
        self.battery_position = battery_position

        # Initialize SAM battery
        self.battery = SAMBattery(
            position=battery_position.copy(),
            total_interceptors=num_interceptors,
            reload_time=reload_time,
            max_simultaneous=max_simultaneous,
            pk_single=pk_single,
            interceptors_remaining=num_interceptors
        )

        # Initialize interceptor statuses
        for i in range(num_interceptors):
            self.battery.interceptors.append(InterceptorStatus(interceptor_id=i))

        # Generate attack wave
        self.targets: List[TargetTrack] = []
        self._generate_attack_wave()

        # Initialize state
        self.state = SaturationState(
            targets_total=num_targets,
            targets_incoming=num_targets,
            interceptors_total=num_interceptors,
            interceptors_remaining=num_interceptors
        )

        # Record initial positions
        for target in self.targets:
            target.record_position()

    def _generate_attack_wave(self):
        """
        Generate incoming cruise missiles for the attack wave (3D).

        Creates targets with:
        - Staggered starting positions (30-second spread)
        - Varying altitudes (10-18 km)
        - Lateral spread in Y direction
        - All heading toward the city
        """
        # Base distance from city (120km)
        base_distance = 120_000
        cruise_speed = 1000.0  # Mach 3

        for i in range(self.num_targets):
            # Stagger arrival: spread over 30 seconds worth of distance
            distance_offset = (i / max(1, self.num_targets - 1)) * 30_000  # 0-30km spread
            missile_distance = base_distance + distance_offset

            # Vary altitude: 10-18 km
            altitude = 10_000 + (i % 4) * 2_000 + random.uniform(-500, 500)

            # Lateral spread in Y direction (±10km for variety)
            lateral_offset_y = random.uniform(-10_000, 10_000)

            # Calculate 3D position
            # City is at (0, 0, earth_radius), missiles approach from -X direction
            missile_angle = missile_distance / self.earth_radius
            missile_radius = self.earth_radius + altitude

            # Position in 3D: coming from negative X, with Y spread
            missile_x = -missile_radius * np.sin(missile_angle)
            missile_y = lateral_offset_y
            missile_z = missile_radius * np.cos(missile_angle)

            position = np.array([missile_x, missile_y, missile_z])

            # Calculate velocity toward city (3D)
            to_city = self.city_position - position
            to_city_unit = to_city / np.linalg.norm(to_city)
            velocity = cruise_speed * to_city_unit

            # Create cruise missile aircraft
            aircraft = Aircraft(
                name=f"CM-{i+1}",
                position=position,
                velocity=velocity,
                max_speed=cruise_speed,
                max_acceleration=15.0,
                max_g=5.0,
                burn_time=300.0,
                drag_coefficient=0.0,
                is_missile=False  # Maintains altitude
            )

            # Create target track
            target = TargetTrack(
                target_id=i,
                aircraft=aircraft
            )

            # Calculate initial time-to-impact
            range_to_city = np.linalg.norm(to_city)
            target.range_to_city = range_to_city
            target.time_to_impact = range_to_city / cruise_speed

            self.targets.append(target)

    def _create_interceptor(self, target: TargetTrack) -> Aircraft:
        """Create an interceptor aircraft aimed at a target."""
        # Calculate where target will be closest to battery (intercept window)
        pos = target.aircraft.position
        vel = target.aircraft.velocity
        to_battery = self.battery_position - pos
        vel_dot = np.dot(vel, vel)

        if vel_dot > 0:
            # Time when target is closest to battery
            t_closest = np.dot(to_battery, vel) / vel_dot
            # Clamp to reasonable time (target might already be past)
            t_closest = max(0, min(t_closest, 120))
            # Add time for SAM to get there (rough estimate)
            closest_pos = pos + vel * t_closest
            range_to_closest = np.linalg.norm(closest_pos - self.battery_position)
            sam_time = range_to_closest / 1700 * 1.2  # 20% margin
            t_intercept = t_closest - sam_time
            t_intercept = max(5, t_intercept)  # At least 5 seconds flight
            predicted_pos = pos + vel * t_intercept
        else:
            # Fallback: aim at current position
            predicted_pos = pos

        launch_dir = predicted_pos - self.battery_position
        launch_dir_unit = launch_dir / np.linalg.norm(launch_dir)

        initial_speed = 500.0

        return Aircraft(
            name=f"SAM",
            position=self.battery_position.copy(),
            velocity=initial_speed * launch_dir_unit,
            max_speed=1700.0,          # PAC-3 spec
            max_acceleration=200.0,
            max_g=60.0,
            burn_time=45.0,
            drag_coefficient=0.0001,
            is_missile=True
        )

    def _prioritize_threats(self):
        """
        Calculate priority for each incoming target.

        Priority is based on:
        - Time to impact (lower = higher priority)
        - Engagement status (unengaged targets get priority boost)
        """
        for target in self.targets:
            if target.status not in ["incoming", "engaged"]:
                target.priority = -1.0
                continue

            # Calculate current time-to-impact
            to_city = self.city_position - target.aircraft.position
            range_to_city = np.linalg.norm(to_city)
            target.range_to_city = range_to_city

            if target.aircraft.speed > 0:
                target.time_to_impact = range_to_city / target.aircraft.speed
            else:
                target.time_to_impact = float('inf')

            # Base priority: inverse of time-to-impact
            target.priority = 100.0 / (target.time_to_impact + 1.0)

            # Boost priority for unengaged targets
            if target.status == "incoming":
                target.priority *= 1.5

    def _assign_interceptors(self):
        """
        Assign available interceptors to highest-priority unengaged targets.
        """
        if not self.battery.can_launch(self.state.time):
            return

        # Get unengaged targets sorted by priority
        unengaged = [t for t in self.targets if t.status == "incoming"]
        unengaged.sort(key=lambda t: t.priority, reverse=True)

        for target in unengaged:
            if not self.battery.can_launch(self.state.time):
                break

            # Find a ready interceptor
            for interceptor in self.battery.interceptors:
                if interceptor.status != "ready":
                    continue

                # Launch this interceptor at the target
                self._launch_interceptor(interceptor, target)
                break

    def _launch_interceptor(self, interceptor: InterceptorStatus, target: TargetTrack):
        """Launch an interceptor at a target."""
        interceptor.aircraft = self._create_interceptor(target)
        interceptor.status = "in_flight"
        interceptor.assigned_target_id = target.target_id
        interceptor.launch_time = self.state.time

        # Initialize guidance systems
        interceptor.seeker = Seeker(
            acquisition_cone=30.0,
            gimbal_limit=45.0,
            max_range=20_000.0,
            lock_time=0.5
        )
        interceptor.guidance = GuidanceController(
            boost_duration=3.0,
            terminal_range=15_000.0,
            N_midcourse=3.0,
            N_terminal=4.0
        )
        interceptor.target_estimator = TargetEstimator(
            process_noise=50.0,
            measurement_noise=100.0
        )

        # Update target status
        target.status = "engaged"
        target.assigned_interceptor_id = interceptor.interceptor_id

        # Update battery state
        self.battery.last_launch_time = self.state.time
        self.battery.interceptors_remaining -= 1
        self.battery.interceptors_in_flight += 1

        # Update simulation state
        self.state.interceptors_launched += 1
        self.state.interceptors_remaining -= 1
        self.state.interceptors_in_flight += 1
        self.state.targets_incoming -= 1

        # Log event
        self.state.events.append((
            self.state.time, "launch", target.target_id, interceptor.interceptor_id
        ))

        print(f"  >>> SAM-{interceptor.interceptor_id} LAUNCHED at CM-{target.target_id+1} "
              f"(t={self.state.time:.1f}s, TTI={target.time_to_impact:.1f}s)")

    def _get_gravity_vector(self, position: np.ndarray) -> np.ndarray:
        """Calculate gravity vector toward Earth's center (3D)."""
        distance = np.linalg.norm(position)
        if distance < 1.0:
            return np.array([0.0, 0.0, 0.0])
        return -self.GRAVITY * position / distance

    def _rotate_vector_toward(self, current_dir: np.ndarray, target_dir: np.ndarray,
                              max_angle: float) -> np.ndarray:
        """
        Rotate current_dir toward target_dir by at most max_angle radians (3D).
        Uses Rodrigues' rotation formula.
        """
        # Compute rotation axis (cross product)
        axis = np.cross(current_dir, target_dir)
        axis_mag = np.linalg.norm(axis)

        if axis_mag < 1e-10:
            # Vectors are parallel (or anti-parallel)
            return current_dir.copy()

        axis = axis / axis_mag

        # Compute angle between vectors
        cos_angle = np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Clamp to maximum turn
        actual_angle = min(angle, max_angle)

        # Rodrigues' rotation formula: v_rot = v*cos(a) + (k x v)*sin(a) + k*(k·v)*(1-cos(a))
        cos_a, sin_a = np.cos(actual_angle), np.sin(actual_angle)
        rotated = (current_dir * cos_a +
                   np.cross(axis, current_dir) * sin_a +
                   axis * np.dot(axis, current_dir) * (1 - cos_a))

        # Ensure unit vector
        norm = np.linalg.norm(rotated)
        if norm > 0:
            return rotated / norm
        return current_dir.copy()

    def _update_interceptor(self, interceptor: InterceptorStatus):
        """Update a single in-flight interceptor."""
        if interceptor.status != "in_flight" or interceptor.aircraft is None:
            return

        target = self.targets[interceptor.assigned_target_id]
        if target.status not in ["engaged", "incoming"]:
            # Target already destroyed or leaked
            interceptor.status = "expended"
            self.battery.interceptors_in_flight -= 1
            self.state.interceptors_in_flight -= 1
            return

        aircraft = interceptor.aircraft
        aircraft.time_since_launch += self.dt

        # Update burn status
        if aircraft.time_since_launch >= aircraft.burn_time:
            aircraft.is_burning = False

        # Update seeker
        interceptor.seeker.update(
            aircraft.position,
            aircraft.velocity,
            aircraft.heading,
            target.aircraft.position,
            target.aircraft.velocity,
            self.dt
        )

        # Update guidance phase
        range_to_target = np.linalg.norm(target.aircraft.position - aircraft.position)
        interceptor.guidance.update_phase(
            aircraft.time_since_launch,
            range_to_target,
            interceptor.seeker
        )

        # Calculate thrust direction based on guidance phase
        phase = interceptor.guidance.current_phase

        if phase == GuidancePhase.BOOST:
            thrust_dir = self._boost_guidance(interceptor, target)
        elif phase == GuidancePhase.MIDCOURSE:
            thrust_dir = self._midcourse_guidance(interceptor, target)
        else:  # TERMINAL
            thrust_dir = self._terminal_guidance(interceptor, target)

        # Apply gravity compensation to guidance direction
        thrust_dir = self._apply_gravity_compensation(aircraft, thrust_dir, target)
        thrust_dir_norm = thrust_dir / np.linalg.norm(thrust_dir)

        # Apply physics with proper steering (3D rotation)
        speed = np.linalg.norm(aircraft.velocity)

        # For missiles, rotate velocity toward thrust direction with g-limited turn rate
        if aircraft.is_missile and speed > 10:
            current_dir = aircraft.velocity / speed

            # Turn rate based on g-limit: omega = a_centripetal / v
            max_turn_rate = (aircraft.max_g * self.GRAVITY) / speed
            max_turn = max_turn_rate * self.dt

            # Use 3D Rodrigues rotation
            new_dir = self._rotate_vector_toward(current_dir, thrust_dir_norm, max_turn)
            aircraft.velocity = speed * new_dir

        # Apply thrust for speed changes (3D)
        acceleration = np.array([0.0, 0.0, 0.0])
        if aircraft.is_missile:
            acceleration += self._get_gravity_vector(aircraft.position)

        thrust_dir_norm = thrust_dir / np.linalg.norm(thrust_dir)
        if aircraft.is_burning:
            acceleration += aircraft.max_acceleration * thrust_dir_norm
        else:
            acceleration += aircraft.max_acceleration * 0.1 * thrust_dir_norm

        aircraft.velocity += acceleration * self.dt

        # Limit speed to max_speed
        speed = np.linalg.norm(aircraft.velocity)
        if speed > aircraft.max_speed:
            aircraft.velocity = aircraft.velocity * (aircraft.max_speed / speed)

        aircraft.position += aircraft.velocity * self.dt

        # Record position
        interceptor.record_position()

        # Check for intercept
        result = self._check_intercept(interceptor, target)

        if result == InterceptResult.HIT:
            self._handle_hit(interceptor, target)
        elif result == InterceptResult.MISS:
            self._handle_miss(interceptor, target)

    def _boost_guidance(self, interceptor: InterceptorStatus, target: TargetTrack) -> np.ndarray:
        """Boost phase guidance - fixed attitude toward predicted intercept."""
        aircraft = interceptor.aircraft
        t_intercept = calculate_intercept_time(aircraft, target.aircraft)

        if t_intercept and t_intercept > 0:
            intercept_point = target.aircraft.position_at_time(t_intercept)
            direction = intercept_point - aircraft.position
        else:
            direction = target.aircraft.position - aircraft.position

        direction = direction / np.linalg.norm(direction)

        # Add climb bias
        up = aircraft.position / np.linalg.norm(aircraft.position)
        thrust_dir = 0.85 * direction + 0.15 * up
        return thrust_dir / np.linalg.norm(thrust_dir)

    def _midcourse_guidance(self, interceptor: InterceptorStatus, target: TargetTrack) -> np.ndarray:
        """Midcourse guidance - lead pursuit toward predicted intercept."""
        aircraft = interceptor.aircraft

        # Use the standard intercept time calculation
        t_intercept = calculate_intercept_time(aircraft, target.aircraft, use_current_speed=True)

        if t_intercept and t_intercept > 0 and t_intercept < 200:
            # Aim at predicted intercept point
            aim_point = target.aircraft.position + target.aircraft.velocity * t_intercept
        else:
            # Fallback: aim directly at target with lead
            to_target = target.aircraft.position - aircraft.position
            range_to_target = np.linalg.norm(to_target)
            t_lead = range_to_target / (aircraft.max_speed + 100)
            aim_point = target.aircraft.position + target.aircraft.velocity * t_lead

        direction = aim_point - aircraft.position
        if np.linalg.norm(direction) > 10:
            direction = direction / np.linalg.norm(direction)

        # Activate seeker if off
        if interceptor.seeker.state == SeekerState.OFF:
            interceptor.seeker.activate()

        return direction

    def _terminal_guidance(self, interceptor: InterceptorStatus, target: TargetTrack) -> np.ndarray:
        """Terminal guidance - Lead pursuit toward predicted intercept point."""
        aircraft = interceptor.aircraft
        seeker = interceptor.seeker

        to_target = target.aircraft.position - aircraft.position
        range_to_target = np.linalg.norm(to_target)

        if range_to_target < 10:
            if aircraft.speed > 10:
                return aircraft.velocity / aircraft.speed
            return np.array([1.0, 0.0, 0.0])

        # Calculate time to intercept
        Vc = seeker.closing_velocity
        if Vc > 50:
            t_intercept = range_to_target / Vc
        else:
            t_intercept = range_to_target / (aircraft.speed + 100)

        # Clamp intercept time
        t_intercept = min(t_intercept, 60.0)

        # Predict where target will be
        aim_point = target.aircraft.position + target.aircraft.velocity * t_intercept

        # Point at aim point
        direction = aim_point - aircraft.position
        if np.linalg.norm(direction) > 10:
            direction = direction / np.linalg.norm(direction)

        return direction

    def _apply_turn_rate_limit(self, aircraft: Aircraft, desired_direction: np.ndarray) -> np.ndarray:
        """Limit how fast the missile can change direction based on g-limits."""
        # Current heading
        current_heading = aircraft.heading

        # Desired heading
        desired_heading = np.degrees(np.arctan2(desired_direction[1], desired_direction[0]))

        # Heading difference
        heading_diff = desired_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # Calculate max turn rate based on g-limits
        speed = aircraft.speed
        if speed < 10:
            max_turn_rate = 30.0  # Base turn rate at low speed
        else:
            max_centripetal = aircraft.max_g * self.GRAVITY
            max_turn_rate_rad = max_centripetal / speed
            max_turn_rate = min(30.0, np.degrees(max_turn_rate_rad))

        max_turn = max_turn_rate * self.dt
        actual_turn = np.clip(heading_diff, -max_turn, max_turn)
        new_heading = current_heading + actual_turn
        new_heading_rad = np.radians(new_heading)

        return np.array([np.cos(new_heading_rad), np.sin(new_heading_rad)])

    def _apply_gravity_compensation(self, aircraft: Aircraft, thrust_dir: np.ndarray,
                                     target: TargetTrack) -> np.ndarray:
        """Apply gravity compensation to thrust direction."""
        if not aircraft.is_missile:
            return thrust_dir

        gravity = self._get_gravity_vector(aircraft.position)
        up = -gravity / np.linalg.norm(gravity)

        to_target = target.aircraft.position - aircraft.position
        range_to_target = np.linalg.norm(to_target)

        if range_to_target > 100:
            vertical = np.dot(to_target, up)
            climb_ratio = vertical / range_to_target
            weight = max(0, climb_ratio) * 0.08 + 0.05

            thrust_dir = thrust_dir + up * weight
            thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)

        return thrust_dir

    def _check_intercept(self, interceptor: InterceptorStatus, target: TargetTrack) -> InterceptResult:
        """Check if interceptor has reached the target."""
        range_to_target = np.linalg.norm(
            target.aircraft.position - interceptor.aircraft.position
        )

        # Track minimum range for debugging
        if not hasattr(interceptor, 'min_range'):
            interceptor.min_range = float('inf')
        if range_to_target < interceptor.min_range:
            interceptor.min_range = range_to_target

        if range_to_target <= self.intercept_range:
            # Roll against Pk
            if random.random() < self.battery.pk_single:
                return InterceptResult.HIT
            else:
                return InterceptResult.MISS

        # Check if interceptor flew past (range increasing and was very close)
        if len(interceptor.path_history) > 20:
            prev_pos = interceptor.path_history[-10]
            prev_range = np.linalg.norm(target.aircraft.position - prev_pos)
            # Only count as miss if was within 2km and now diverging
            if range_to_target > prev_range * 1.1 and interceptor.min_range < 2000:
                return InterceptResult.MISS

        return InterceptResult.IN_PROGRESS

    def _handle_hit(self, interceptor: InterceptorStatus, target: TargetTrack):
        """Handle successful intercept."""
        interceptor.status = "hit"
        target.status = "destroyed"

        self.battery.interceptors_in_flight -= 1
        self.state.interceptors_in_flight -= 1
        self.state.targets_destroyed += 1

        self.state.events.append((
            self.state.time, "hit", target.target_id, interceptor.interceptor_id
        ))

        print(f"  >>> HIT! SAM-{interceptor.interceptor_id} destroyed CM-{target.target_id+1} "
              f"at t={self.state.time:.1f}s")

    def _handle_miss(self, interceptor: InterceptorStatus, target: TargetTrack):
        """Handle missed intercept."""
        interceptor.status = "miss"
        target.status = "incoming"  # Back to incoming, can be re-engaged
        target.assigned_interceptor_id = None

        self.battery.interceptors_in_flight -= 1
        self.state.interceptors_in_flight -= 1
        self.state.targets_incoming += 1

        self.state.events.append((
            self.state.time, "miss", target.target_id, interceptor.interceptor_id
        ))

        print(f"  >>> MISS! SAM-{interceptor.interceptor_id} missed CM-{target.target_id+1} "
              f"at t={self.state.time:.1f}s")

    def _update_targets(self):
        """Update all target positions and check for leakers."""
        for target in self.targets:
            if target.status in ["destroyed", "leaked"]:
                continue

            # Update position
            target.aircraft.position += target.aircraft.velocity * self.dt
            target.record_position()

            # Check if target reached city (leaked)
            to_city = self.city_position - target.aircraft.position
            range_to_city = np.linalg.norm(to_city)

            if range_to_city < 5000:  # Within 5km of city = impact
                target.status = "leaked"
                self.state.targets_leaked += 1

                if target.assigned_interceptor_id is not None:
                    # Clear the engagement
                    for interceptor in self.battery.interceptors:
                        if interceptor.interceptor_id == target.assigned_interceptor_id:
                            if interceptor.status == "in_flight":
                                interceptor.status = "expended"
                                self.battery.interceptors_in_flight -= 1
                                self.state.interceptors_in_flight -= 1
                else:
                    self.state.targets_incoming -= 1

                self.state.events.append((
                    self.state.time, "leak", target.target_id, None
                ))

                print(f"  >>> LEAK! CM-{target.target_id+1} reached city at t={self.state.time:.1f}s")

    def _check_termination(self) -> bool:
        """Check if simulation should end."""
        # All targets resolved
        active_targets = sum(1 for t in self.targets if t.status in ["incoming", "engaged"])
        if active_targets == 0:
            self.state.status = "completed"
            return True

        # Timeout
        if self.state.time >= self.max_time:
            self.state.status = "timeout"
            return True

        # Out of interceptors with targets still incoming
        if (self.battery.interceptors_remaining == 0 and
            self.state.interceptors_in_flight == 0 and
            any(t.status == "incoming" for t in self.targets)):
            # Wait for remaining engaged targets to resolve
            engaged = sum(1 for t in self.targets if t.status == "engaged")
            if engaged == 0:
                self.state.status = "completed"
                return True

        return False

    def step(self) -> bool:
        """Execute one simulation step."""
        if self.state.status != "running":
            return False

        # 1. Update target positions and check leakers
        self._update_targets()

        # 2. Prioritize threats
        self._prioritize_threats()

        # 3. Assign new interceptors to high-priority targets
        self._assign_interceptors()

        # 4. Update all in-flight interceptors
        for interceptor in self.battery.interceptors:
            self._update_interceptor(interceptor)

        # 5. Advance time
        self.state.time += self.dt

        # 6. Check termination
        if self._check_termination():
            return False

        return True

    def run(self, verbose: bool = True) -> SaturationState:
        """Run the complete simulation."""
        if verbose:
            print(f"\n{'='*60}")
            print("SATURATION ATTACK SIMULATION")
            print('='*60)
            print(f"Targets: {self.num_targets} cruise missiles")
            print(f"Interceptors: {self.num_interceptors} SAMs (Pk={self.battery.pk_single:.0%})")
            print(f"Reload time: {self.battery.reload_time:.1f}s")
            print(f"Max simultaneous: {self.battery.max_simultaneous}")
            print(f"Intercept range: {self.intercept_range}m")
            print('='*60)

        step_count = 0
        while self.step():
            step_count += 1

            # Progress every 10 seconds
            if verbose and step_count % int(10.0 / self.dt) == 0:
                active = sum(1 for t in self.targets if t.status in ["incoming", "engaged"])
                print(f"  t={self.state.time:.1f}s: {active} active targets, "
                      f"{self.state.interceptors_in_flight} SAMs in flight")

        if verbose:
            print(f"\n{'='*60}")
            print("SIMULATION COMPLETE")
            print('='*60)
            print(f"Duration: {self.state.time:.1f}s")
            print(f"Targets destroyed: {self.state.targets_destroyed}/{self.num_targets}")
            print(f"Targets leaked: {self.state.targets_leaked}/{self.num_targets}")
            print(f"Interceptors used: {self.state.interceptors_launched}/{self.num_interceptors}")

            # Calculate defense effectiveness
            if self.num_targets > 0:
                effectiveness = self.state.targets_destroyed / self.num_targets * 100
                print(f"Defense effectiveness: {effectiveness:.1f}%")

        return self.state


def plot_simulation_result(sim: InterceptSimulation, save_path: str = None):
    """
    Plot the results of a completed simulation.

    Shows:
    - Pursuer and target paths
    - Start and end positions
    - Range over time subplot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Trajectory
    ax1 = axes[0]

    pursuer_path = np.array(sim.state.pursuer_path)
    target_path = np.array(sim.state.target_path)

    # Plot paths
    ax1.plot(pursuer_path[:, 0], pursuer_path[:, 1], 'b-', linewidth=2,
             label=f'Pursuer ({sim.pursuer.name})', alpha=0.8)
    ax1.plot(target_path[:, 0], target_path[:, 1], 'r-', linewidth=2,
             label=f'Target ({sim.target.name})', alpha=0.8)

    # Mark start positions
    ax1.plot(pursuer_path[0, 0], pursuer_path[0, 1], 'bo', markersize=12,
             label='Pursuer start')
    ax1.plot(target_path[0, 0], target_path[0, 1], 'ro', markersize=12,
             label='Target start')

    # Mark end positions
    ax1.plot(pursuer_path[-1, 0], pursuer_path[-1, 1], 'b^', markersize=15)
    ax1.plot(target_path[-1, 0], target_path[-1, 1], 'r^', markersize=15)

    # Mark intercept if successful
    if sim.state.status == "intercepted":
        ax1.plot(pursuer_path[-1, 0], pursuer_path[-1, 1], 'g*', markersize=25,
                 label='INTERCEPT', zorder=10)

    ax1.set_xlabel('X Position (meters)', fontsize=12)
    ax1.set_ylabel('Y Position (meters)', fontsize=12)
    ax1.set_title(f'Intercept Simulation - {sim.state.status.upper()}', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right plot: Range over time
    ax2 = axes[1]

    ax2.plot(sim.state.time_history, sim.state.range_history, 'g-', linewidth=2)
    ax2.axhline(y=sim.intercept_range, color='r', linestyle='--',
                label=f'Intercept range ({sim.intercept_range}m)')

    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Range (meters)', fontsize=12)
    ax2.set_title('Range to Target Over Time', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add result annotation
    result_text = f"Result: {sim.state.status.upper()}\n"
    result_text += f"Duration: {sim.state.time:.2f}s\n"
    result_text += f"Final range: {sim.state.range_history[-1]:.1f}m"

    ax2.text(0.98, 0.98, result_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
    return fig


def animate_simulation(sim: InterceptSimulation, interval: int = 50,
                       save_path: str = None):
    """
    Create an animated visualization of the simulation with aircraft shapes.

    Args:
        sim: A completed InterceptSimulation
        interval: Milliseconds between frames
        save_path: Optional path to save animation (requires ffmpeg)
    """
    from matplotlib.patches import Polygon, Circle

    fig, ax = plt.subplots(figsize=(14, 10))

    pursuer_path = np.array(sim.state.pursuer_path)
    target_path = np.array(sim.state.target_path)

    # Set up plot bounds (scale margin based on plot size)
    all_x = np.concatenate([pursuer_path[:, 0], target_path[:, 0]])
    all_y = np.concatenate([pursuer_path[:, 1], target_path[:, 1]])
    plot_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    margin = plot_range * 0.1  # 10% margin
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Calculate heading from path
    def get_heading(path, frame):
        if frame < len(path) - 1:
            delta = path[frame + 1] - path[frame]
        else:
            delta = path[frame] - path[frame - 1]
        return np.degrees(np.arctan2(delta[1], delta[0]))

    # Scale shapes based on plot size
    aircraft_size = plot_range * 0.02

    # Aircraft shape (triangle)
    def create_aircraft_shape(size):
        return np.array([
            [size, 0],                    # Nose
            [-size * 0.6, size * 0.4],    # Left wing
            [-size * 0.3, 0],             # Tail center
            [-size * 0.6, -size * 0.4],   # Right wing
        ])

    # Missile shape
    def create_missile_shape(size):
        return np.array([
            [size, 0],                    # Nose
            [size * 0.3, size * 0.12],
            [-size * 0.5, size * 0.12],
            [-size * 0.7, size * 0.25],   # Left fin
            [-size * 0.5, size * 0.08],
            [-size * 0.5, -size * 0.08],
            [-size * 0.7, -size * 0.25],  # Right fin
            [-size * 0.5, -size * 0.12],
            [size * 0.3, -size * 0.12],
        ])

    def rotate_shape(shape, angle_deg, position):
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = shape @ rotation_matrix.T
        return rotated + position

    # Draw ground line and terrain
    ground_y = pursuer_path[0, 1]  # Ground level is where missile starts
    ax.axhline(y=ground_y, color='saddlebrown', linewidth=3, alpha=0.7, label='Ground')
    ax.fill_between([all_x.min() - margin, all_x.max() + margin],
                    ground_y - plot_range * 0.15, ground_y, color='saddlebrown', alpha=0.2)

    # Ground station marker (scaled)
    station_x, station_y = pursuer_path[0]
    station_size = max(12, min(20, plot_range / 3000))  # Scale marker size
    ax.plot(station_x, station_y, 's', color='darkgreen', markersize=station_size,
            zorder=5, label='Launch Site')
    ax.annotate('SAM SITE', (station_x, station_y),
                textcoords="offset points", xytext=(15, -25),
                fontsize=10, fontweight='bold', color='darkgreen')

    # Initialize trails
    pursuer_trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.4, label='Missile path')
    target_trail, = ax.plot([], [], 'r-', linewidth=2, alpha=0.4, label='Target path')

    # Create shape templates
    pursuer_shape = create_missile_shape(aircraft_size * 0.8)
    target_shape = create_aircraft_shape(aircraft_size)

    # Create aircraft polygons
    pursuer_poly = Polygon(pursuer_shape, closed=True, fc='blue', ec='darkblue',
                           linewidth=2, zorder=10)
    target_poly = Polygon(target_shape, closed=True, fc='red', ec='darkred',
                          linewidth=2, zorder=10)
    ax.add_patch(pursuer_poly)
    ax.add_patch(target_poly)

    # Evasion radius circle
    evasion_circle = Circle((0, 0), sim.evasion_radius, fill=False,
                            color='orange', linestyle='--', linewidth=1.5,
                            alpha=0.6, label=f'Evasion radius ({sim.evasion_radius:.0f}m)')
    ax.add_patch(evasion_circle)

    # Info display
    info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Status text
    status_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=14,
                          verticalalignment='top', horizontalalignment='right',
                          fontweight='bold')

    ax.set_xlabel('Horizontal Distance (meters)', fontsize=12)
    ax.set_ylabel('Altitude (meters)', fontsize=12)
    ax.set_title('Surface-to-Air Missile Intercept', fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f8ff')

    def init():
        pursuer_trail.set_data([], [])
        target_trail.set_data([], [])
        pursuer_poly.set_xy(pursuer_shape)
        target_poly.set_xy(target_shape)
        evasion_circle.center = (0, 0)
        info_box.set_text('')
        status_text.set_text('')
        return pursuer_trail, target_trail, pursuer_poly, target_poly, evasion_circle, info_box, status_text

    def animate(frame):
        # Update trails
        pursuer_trail.set_data(pursuer_path[:frame+1, 0], pursuer_path[:frame+1, 1])
        target_trail.set_data(target_path[:frame+1, 0], target_path[:frame+1, 1])

        # Get positions and headings
        pursuer_pos = pursuer_path[frame]
        target_pos = target_path[frame]
        pursuer_heading = get_heading(pursuer_path, frame)
        target_heading = get_heading(target_path, frame)

        # Rotate and position aircraft
        pursuer_poly.set_xy(rotate_shape(pursuer_shape, pursuer_heading, pursuer_pos))
        target_poly.set_xy(rotate_shape(target_shape, target_heading, target_pos))

        # Update evasion circle
        evasion_circle.center = (target_pos[0], target_pos[1])

        # Check evasion mode
        current_range = sim.state.range_history[frame]
        in_evasion = current_range <= sim.evasion_radius

        if in_evasion:
            evasion_circle.set_color('red')
            evasion_circle.set_linewidth(2.5)
            mode_str = "EVADING"
        else:
            evasion_circle.set_color('orange')
            evasion_circle.set_linewidth(1.5)
            mode_str = "Cruising"

        # Update info
        missile_alt = pursuer_pos[1] / 1000  # km
        target_alt = target_pos[1] / 1000    # km
        info_str = f"Time:    {sim.state.time_history[frame]:5.1f}s\n"
        info_str += f"Range:   {current_range/1000:5.1f}km\n"
        info_str += f"SAM Alt: {missile_alt:5.1f}km\n"
        info_str += f"Tgt Alt: {target_alt:5.1f}km\n"
        info_str += f"Status:  {mode_str}"
        info_box.set_text(info_str)

        # Final status
        if frame == len(pursuer_path) - 1:
            if sim.state.status == "intercepted":
                status_text.set_text("INTERCEPTED!")
                status_text.set_color('green')
            elif sim.state.status == "escaped":
                status_text.set_text("TARGET ESCAPED")
                status_text.set_color('red')
            elif sim.state.status == "timeout":
                status_text.set_text("TIMEOUT")
                status_text.set_color('orange')
        else:
            status_text.set_text("")

        return pursuer_trail, target_trail, pursuer_poly, target_poly, evasion_circle, info_box, status_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(pursuer_path), interval=interval,
                         blit=True, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=30)
        print("Done!")

    plt.tight_layout()
    plt.show()
    return anim


def animate_simulation_earth(sim: InterceptSimulation, interval: int = 50,
                              save_path: str = None):
    """
    Create an animated visualization showing Earth's curvature.

    The SAM and cruise missile are shown on a curved Earth surface,
    with the city being defended visible on the surface.

    Args:
        sim: A completed InterceptSimulation with earth_radius and city_position attributes
        interval: Milliseconds between frames
        save_path: Optional path to save animation
    """
    from matplotlib.patches import Polygon, Circle

    fig, ax = plt.subplots(figsize=(16, 10))

    pursuer_path = np.array(sim.state.pursuer_path)
    target_path = np.array(sim.state.target_path)

    # Get Earth parameters from simulation
    earth_radius = getattr(sim, 'earth_radius', 6_371_000)
    city_pos = getattr(sim, 'city_position', np.array([0, earth_radius]))

    # Calculate view bounds - we want to see the relevant portion of Earth
    all_x = np.concatenate([pursuer_path[:, 0], target_path[:, 0], [city_pos[0]]])
    all_y = np.concatenate([pursuer_path[:, 1], target_path[:, 1], [city_pos[1]]])

    # Center the view on the action
    center_x = (all_x.min() + all_x.max()) / 2
    center_y = (all_y.min() + all_y.max()) / 2

    # Calculate view range (with margin)
    range_x = all_x.max() - all_x.min()
    range_y = all_y.max() - all_y.min()
    view_range = max(range_x, range_y) * 1.3

    ax.set_xlim(center_x - view_range/2, center_x + view_range/2)
    ax.set_ylim(center_y - view_range/2, center_y + view_range/2)

    # Draw Earth's surface as a circle (Earth's center is at origin)
    earth_surface = Circle((0, 0), earth_radius, fill=True,
                           facecolor='#2d5016', edgecolor='#1a3009',
                           linewidth=3, zorder=1)
    ax.add_patch(earth_surface)

    # Add atmosphere layer (100km above Earth)
    atmosphere_outer = earth_radius + 100_000
    atmosphere = Circle((0, 0), atmosphere_outer, fill=True,
                        facecolor='#87ceeb', edgecolor='none',
                        alpha=0.15, zorder=0)
    ax.add_patch(atmosphere)

    # Add space background
    ax.set_facecolor('#0a0a1a')

    # Mark the city on Earth's surface
    city_marker_size = max(8, min(15, view_range / 50000))
    ax.plot(city_pos[0], city_pos[1], 'o', color='yellow', markersize=city_marker_size,
            zorder=15, label='City (Target)')
    ax.annotate('CITY', (city_pos[0], city_pos[1]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=10, fontweight='bold', color='yellow',
                zorder=15)

    # Mark SAM launch site
    sam_start = pursuer_path[0]
    ax.plot(sam_start[0], sam_start[1], 's', color='lime', markersize=city_marker_size * 0.8,
            zorder=15, label='SAM Site')
    ax.annotate('SAM', (sam_start[0], sam_start[1]),
                textcoords="offset points", xytext=(10, -15),
                fontsize=9, fontweight='bold', color='lime',
                zorder=15)

    # Calculate heading from path
    def get_heading(path, frame):
        if frame < len(path) - 1:
            delta = path[frame + 1] - path[frame]
        else:
            delta = path[frame] - path[frame - 1]
        return np.degrees(np.arctan2(delta[1], delta[0]))

    # Scale shapes based on view
    shape_size = view_range * 0.015

    # SAM interceptor shape (sleek missile)
    def create_sam_shape(size):
        return np.array([
            [size, 0],                    # Nose
            [size * 0.3, size * 0.08],
            [-size * 0.4, size * 0.08],
            [-size * 0.6, size * 0.2],    # Fin
            [-size * 0.4, size * 0.05],
            [-size * 0.4, -size * 0.05],
            [-size * 0.6, -size * 0.2],   # Fin
            [-size * 0.4, -size * 0.08],
            [size * 0.3, -size * 0.08],
        ])

    # Cruise missile shape (larger, with wings)
    def create_cruise_missile_shape(size):
        return np.array([
            [size, 0],                    # Nose
            [size * 0.4, size * 0.1],
            [0, size * 0.1],
            [-size * 0.2, size * 0.4],    # Wing
            [-size * 0.3, size * 0.1],
            [-size * 0.6, size * 0.1],
            [-size * 0.7, size * 0.25],   # Tail
            [-size * 0.6, size * 0.05],
            [-size * 0.6, -size * 0.05],
            [-size * 0.7, -size * 0.25],  # Tail
            [-size * 0.6, -size * 0.1],
            [-size * 0.3, -size * 0.1],
            [-size * 0.2, -size * 0.4],   # Wing
            [0, -size * 0.1],
            [size * 0.4, -size * 0.1],
        ])

    def rotate_shape(shape, angle_deg, position):
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = shape @ rotation_matrix.T
        return rotated + position

    # Initialize trails
    pursuer_trail, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.6, label='SAM path')
    target_trail, = ax.plot([], [], 'red', linewidth=2, alpha=0.6, label='Cruise missile path')

    # Create shape templates
    pursuer_shape = create_sam_shape(shape_size * 0.7)
    target_shape = create_cruise_missile_shape(shape_size)

    # Create missile polygons
    pursuer_poly = Polygon(pursuer_shape, closed=True, fc='cyan', ec='white',
                           linewidth=1.5, zorder=20)
    target_poly = Polygon(target_shape, closed=True, fc='red', ec='darkred',
                          linewidth=1.5, zorder=20)
    ax.add_patch(pursuer_poly)
    ax.add_patch(target_poly)

    # Evasion radius circle
    evasion_circle = Circle((0, 0), sim.evasion_radius, fill=False,
                            color='orange', linestyle='--', linewidth=1.5,
                            alpha=0.5, label=f'Evasion radius ({sim.evasion_radius/1000:.0f}km)')
    ax.add_patch(evasion_circle)

    # Seeker cone (Wedge) - shows acquisition/tracking cone
    seeker_cone_size = view_range * 0.15  # Visual size of the cone
    seeker_wedge = Wedge((0, 0), seeker_cone_size,
                         -sim.seeker.acquisition_cone,  # Start angle (relative)
                         sim.seeker.acquisition_cone,   # End angle (relative)
                         width=seeker_cone_size * 0.1,  # Thin wedge
                         facecolor='lime', edgecolor='white',
                         alpha=0.0, linewidth=1, zorder=18)
    ax.add_patch(seeker_wedge)

    # Radar range circle (centered on SAM site)
    radar_range = getattr(sim, 'radar_range', 400_000)
    radar_circle = Circle((sam_start[0], sam_start[1]), radar_range, fill=False,
                          color='#00ff00', linestyle=':', linewidth=2,
                          alpha=0.4, label=f'Radar range ({radar_range/1000:.0f}km)')
    ax.add_patch(radar_circle)

    # Radar status display (right side)
    radar_box = ax.text(0.98, 0.75, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='right',
                        family='monospace', color='lime',
                        bbox=dict(boxstyle='round', facecolor='#0a1a0a', alpha=0.9, edgecolor='lime'))

    # Info display (dark themed)
    info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', family='monospace', color='white',
                       bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9, edgecolor='cyan'))

    # Status text
    status_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=16,
                          verticalalignment='top', horizontalalignment='right',
                          fontweight='bold')

    # Range line between missiles
    range_line, = ax.plot([], [], 'yellow', linewidth=1, linestyle=':', alpha=0.7)

    ax.set_xlabel('Distance (meters)', fontsize=12, color='white')
    ax.set_ylabel('Distance (meters)', fontsize=12, color='white')
    ax.set_title('Global Missile Defense - Earth Curvature View', fontsize=14,
                 fontweight='bold', color='white')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=9,
              facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    def init():
        pursuer_trail.set_data([], [])
        target_trail.set_data([], [])
        pursuer_poly.set_xy(pursuer_shape)
        target_poly.set_xy(target_shape)
        evasion_circle.center = (0, 0)
        seeker_wedge.set_center((0, 0))
        seeker_wedge.set_alpha(0.0)
        range_line.set_data([], [])
        info_box.set_text('')
        status_text.set_text('')
        radar_box.set_text('')
        return (pursuer_trail, target_trail, pursuer_poly, target_poly,
                evasion_circle, seeker_wedge, range_line, info_box, status_text, radar_box)

    def animate(frame):
        # Get radar status for this frame
        radar_status = sim.state.radar_status_history[frame] if frame < len(sim.state.radar_status_history) else "launched"

        # Update trails - only show SAM trail after launch
        if radar_status == "launched":
            # Find the launch frame
            launch_frame = 0
            for i, status in enumerate(sim.state.radar_status_history):
                if status == "launched":
                    launch_frame = i
                    break
            pursuer_trail.set_data(pursuer_path[launch_frame:frame+1, 0], pursuer_path[launch_frame:frame+1, 1])
        else:
            pursuer_trail.set_data([], [])

        target_trail.set_data(target_path[:frame+1, 0], target_path[:frame+1, 1])

        # Get positions and headings
        pursuer_pos = pursuer_path[frame]
        target_pos = target_path[frame]
        pursuer_heading = get_heading(pursuer_path, frame)
        target_heading = get_heading(target_path, frame)

        # Only show SAM missile after launch
        if radar_status == "launched":
            pursuer_poly.set_xy(rotate_shape(pursuer_shape, pursuer_heading, pursuer_pos))
            pursuer_poly.set_alpha(1.0)

            # Update seeker cone position and orientation
            seeker_wedge.set_center(pursuer_pos)
            seeker_wedge.set_theta1(pursuer_heading - sim.seeker.acquisition_cone)
            seeker_wedge.set_theta2(pursuer_heading + sim.seeker.acquisition_cone)

            # Get seeker state for this frame and set color
            if frame < len(sim.state.seeker_state_history):
                seeker_state = sim.state.seeker_state_history[frame]
                if seeker_state == SeekerState.LOCKED:
                    seeker_wedge.set_facecolor('lime')
                    seeker_wedge.set_alpha(0.3)
                elif seeker_state == SeekerState.SEARCHING:
                    seeker_wedge.set_facecolor('yellow')
                    seeker_wedge.set_alpha(0.2)
                elif seeker_state == SeekerState.LOST:
                    seeker_wedge.set_facecolor('red')
                    seeker_wedge.set_alpha(0.25)
                else:
                    seeker_wedge.set_alpha(0.0)
            else:
                seeker_wedge.set_alpha(0.0)
        else:
            # Hide the SAM (move it off-screen or make transparent)
            pursuer_poly.set_alpha(0.0)
            seeker_wedge.set_alpha(0.0)

        # Always show the target
        target_poly.set_xy(rotate_shape(target_shape, target_heading, target_pos))

        # Update evasion circle (around target)
        evasion_circle.center = (target_pos[0], target_pos[1])

        # Update range line only when SAM is launched
        if radar_status == "launched":
            range_line.set_data([pursuer_pos[0], target_pos[0]],
                               [pursuer_pos[1], target_pos[1]])
        else:
            range_line.set_data([], [])

        # Calculate altitudes (distance from Earth center minus Earth radius)
        pursuer_alt = np.linalg.norm(pursuer_pos) - earth_radius
        target_alt = np.linalg.norm(target_pos) - earth_radius

        # Distance from target to city (along Earth's surface approximation)
        target_to_city = np.linalg.norm(target_pos - city_pos)

        # Update radar status display
        if radar_status == "searching":
            radar_str = "RADAR: SEARCHING\n"
            radar_str += "Status: Scanning..."
            radar_box.set_text(radar_str)
            radar_box.set_color('#888888')
            radar_circle.set_color('#444444')
            radar_circle.set_linestyle(':')
            mode_str = "SEARCHING"
        elif radar_status == "detected":
            radar_str = "RADAR: CONTACT!\n"
            radar_str += "Status: Preparing launch..."
            radar_box.set_text(radar_str)
            radar_box.set_color('yellow')
            radar_circle.set_color('yellow')
            radar_circle.set_linestyle('-')
            mode_str = "DETECTED"
        else:  # launched
            radar_str = "RADAR: TRACKING\n"
            radar_str += "Status: SAM in flight"
            radar_box.set_text(radar_str)
            radar_box.set_color('lime')
            radar_circle.set_color('lime')
            radar_circle.set_linestyle('-')

            # Check evasion mode
            current_range = sim.state.range_history[frame]
            in_evasion = current_range <= sim.evasion_radius

            if in_evasion:
                evasion_circle.set_color('red')
                evasion_circle.set_linewidth(2.5)
                mode_str = "TERMINAL"
            else:
                evasion_circle.set_color('orange')
                evasion_circle.set_linewidth(1.5)
                mode_str = "INTERCEPT"

        # Calculate range for display
        current_range = sim.state.range_history[frame]

        # Get flight dynamics data if available
        pursuer_speed = sim.state.pursuer_speed_history[frame] if frame < len(sim.state.pursuer_speed_history) else 0
        target_speed = sim.state.target_speed_history[frame] if frame < len(sim.state.target_speed_history) else 0
        current_g = sim.state.pursuer_g_history[frame] if frame < len(sim.state.pursuer_g_history) else 0
        is_burning = sim.state.pursuer_burn_history[frame] if frame < len(sim.state.pursuer_burn_history) else True

        # Get guidance state for this frame
        guidance_phase = sim.state.guidance_phase_history[frame] if frame < len(sim.state.guidance_phase_history) else GuidancePhase.BOOST
        seeker_state = sim.state.seeker_state_history[frame] if frame < len(sim.state.seeker_state_history) else SeekerState.OFF
        los_rate = sim.state.los_rate_history[frame] if frame < len(sim.state.los_rate_history) else 0.0
        closing_vel = sim.state.closing_velocity_history[frame] if frame < len(sim.state.closing_velocity_history) else 0.0

        # Update info with flight dynamics
        info_str = f"Time:        {sim.state.time_history[frame]:6.1f}s\n"
        info_str += f"Range:       {current_range/1000:6.1f} km\n"
        if radar_status == "launched":
            info_str += f"SAM Alt:     {pursuer_alt/1000:6.1f} km\n"
            info_str += f"SAM Speed:   {pursuer_speed:6.0f} m/s\n"
            burn_str = "BURN" if is_burning else "COAST"
            info_str += f"SAM Status:  {burn_str} ({current_g:.1f}g)\n"
            # Add guidance info
            info_str += f"Guidance:    {guidance_phase.name}\n"
            info_str += f"Seeker:      {seeker_state.name}\n"
            info_str += f"LOS Rate:    {los_rate*1000:6.2f} mrad/s\n"
            info_str += f"Close Vel:   {closing_vel:6.0f} m/s\n"
        else:
            info_str += f"SAM Alt:     --- (on ground)\n"
            info_str += f"SAM Speed:   --- \n"
            info_str += f"SAM Status:  READY\n"
            info_str += f"Guidance:    STANDBY\n"
            info_str += f"Seeker:      OFF\n"
        info_str += f"Target Alt:  {target_alt/1000:6.1f} km\n"
        info_str += f"Target Spd:  {target_speed:6.0f} m/s\n"
        info_str += f"To City:     {target_to_city/1000:6.1f} km"
        info_box.set_text(info_str)

        # Final status
        if frame == len(pursuer_path) - 1:
            if sim.state.status == "intercepted":
                status_text.set_text("TARGET DESTROYED!")
                status_text.set_color('lime')
            elif sim.state.status == "escaped":
                status_text.set_text("TARGET ESCAPED - CITY HIT!")
                status_text.set_color('red')
            elif sim.state.status == "timeout":
                status_text.set_text("ENGAGEMENT TIMEOUT")
                status_text.set_color('orange')
        else:
            status_text.set_text("")

        return (pursuer_trail, target_trail, pursuer_poly, target_poly,
                evasion_circle, seeker_wedge, range_line, info_box, status_text, radar_box)

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(pursuer_path), interval=interval,
                         blit=True, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=30)
        print("Done!")

    plt.tight_layout()
    plt.show()
    return anim


def animate_saturation_attack(sim: SaturationSimulation, interval: int = 50,
                               save_path: str = None):
    """
    Animated visualization of a saturation attack scenario.

    Shows multiple incoming cruise missiles and SAM interceptors with:
    - Color-coded target trails by status
    - Interceptor trails with guidance phase colors
    - Hit/Miss explosion effects
    - Real-time status panels for threats and battery state

    Args:
        sim: A completed SaturationSimulation
        interval: Milliseconds between frames
        save_path: Optional path to save animation
    """
    from matplotlib.patches import Polygon, Circle

    fig, ax = plt.subplots(figsize=(18, 12))

    # Get Earth parameters
    earth_radius = sim.earth_radius
    city_pos = sim.city_position
    battery_pos = sim.battery_position

    # Collect all positions for view bounds
    all_positions = []
    for target in sim.targets:
        all_positions.extend(target.path_history)
    for interceptor in sim.battery.interceptors:
        all_positions.extend(interceptor.path_history)
    all_positions.append(city_pos)
    all_positions.append(battery_pos)

    all_positions = np.array(all_positions)
    if len(all_positions) == 0:
        print("No data to animate!")
        return None

    # Calculate view bounds
    center_x = (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2
    center_y = (all_positions[:, 1].min() + all_positions[:, 1].max()) / 2
    range_x = all_positions[:, 0].max() - all_positions[:, 0].min()
    range_y = all_positions[:, 1].max() - all_positions[:, 1].min()
    view_range = max(range_x, range_y) * 1.3

    ax.set_xlim(center_x - view_range/2, center_x + view_range/2)
    ax.set_ylim(center_y - view_range/2, center_y + view_range/2)

    # Draw Earth surface
    earth_surface = Circle((0, 0), earth_radius, fill=True,
                           facecolor='#2d5016', edgecolor='#1a3009',
                           linewidth=3, zorder=1)
    ax.add_patch(earth_surface)

    # Atmosphere
    atmosphere = Circle((0, 0), earth_radius + 100_000, fill=True,
                        facecolor='#87ceeb', edgecolor='none',
                        alpha=0.15, zorder=0)
    ax.add_patch(atmosphere)
    ax.set_facecolor('#0a0a1a')

    # Mark city
    marker_size = max(10, min(18, view_range / 40000))
    ax.plot(city_pos[0], city_pos[1], '*', color='yellow', markersize=marker_size * 1.5,
            zorder=25, label='City')
    ax.annotate('CITY', (city_pos[0], city_pos[1]),
                textcoords="offset points", xytext=(12, 12),
                fontsize=11, fontweight='bold', color='yellow', zorder=25)

    # Mark SAM site
    ax.plot(battery_pos[0], battery_pos[1], 's', color='lime', markersize=marker_size,
            zorder=25, label='SAM Battery')
    ax.annotate('SAM', (battery_pos[0], battery_pos[1]),
                textcoords="offset points", xytext=(12, -15),
                fontsize=10, fontweight='bold', color='lime', zorder=25)

    # Shape functions
    shape_size = view_range * 0.012

    def create_cm_shape(size):
        return np.array([
            [size, 0], [size*0.4, size*0.1], [0, size*0.1],
            [-size*0.2, size*0.35], [-size*0.3, size*0.1],
            [-size*0.5, size*0.1], [-size*0.6, size*0.2],
            [-size*0.5, 0], [-size*0.6, -size*0.2],
            [-size*0.5, -size*0.1], [-size*0.3, -size*0.1],
            [-size*0.2, -size*0.35], [0, -size*0.1], [size*0.4, -size*0.1]
        ])

    def create_sam_shape(size):
        return np.array([
            [size*0.8, 0], [size*0.2, size*0.06], [-size*0.3, size*0.06],
            [-size*0.5, size*0.15], [-size*0.3, 0],
            [-size*0.5, -size*0.15], [-size*0.3, -size*0.06],
            [size*0.2, -size*0.06]
        ])

    def rotate_shape(shape, angle_deg, position):
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return shape @ rot.T + position

    def get_heading(path, idx):
        if len(path) < 2:
            return 0
        if idx < len(path) - 1:
            delta = path[idx + 1] - path[idx]
        else:
            delta = path[idx] - path[idx - 1]
        return np.degrees(np.arctan2(delta[1], delta[0]))

    # Color schemes
    target_colors = ['#ff4444', '#ff6644', '#ff8844', '#ffaa44', '#ffcc44', '#ffee44']
    sam_color = '#00ffff'

    # Create polygon objects for each target and interceptor
    target_polys = []
    target_trails = []
    for i, target in enumerate(sim.targets):
        color = target_colors[i % len(target_colors)]
        poly = Polygon(create_cm_shape(shape_size), closed=True,
                      fc=color, ec='white', linewidth=1, zorder=20)
        ax.add_patch(poly)
        target_polys.append(poly)

        trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
        target_trails.append(trail)

    sam_polys = []
    sam_trails = []
    for interceptor in sim.battery.interceptors:
        poly = Polygon(create_sam_shape(shape_size * 0.8), closed=True,
                      fc=sam_color, ec='white', linewidth=1, zorder=21)
        poly.set_alpha(0.0)  # Hidden until launched
        ax.add_patch(poly)
        sam_polys.append(poly)

        trail, = ax.plot([], [], color=sam_color, linewidth=1.5, alpha=0.6)
        sam_trails.append(trail)

    # Event markers (hits/misses/leaks)
    event_markers = []

    # Find max frames needed
    max_frames = max(
        max((len(t.path_history) for t in sim.targets), default=1),
        max((len(i.path_history) for i in sim.battery.interceptors), default=1)
    )

    # Info panels
    threat_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', family='monospace', color='white',
                        bbox=dict(boxstyle='round', facecolor='#1a0a0a', alpha=0.9,
                                 edgecolor='red', linewidth=2))

    battery_box = ax.text(0.02, 0.70, '', transform=ax.transAxes, fontsize=11,
                         verticalalignment='top', family='monospace', color='white',
                         bbox=dict(boxstyle='round', facecolor='#0a1a0a', alpha=0.9,
                                  edgecolor='lime', linewidth=2))

    time_box = ax.text(0.5, 0.98, '', transform=ax.transAxes, fontsize=14,
                      verticalalignment='top', horizontalalignment='center',
                      fontweight='bold', color='white',
                      bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9,
                               edgecolor='cyan'))

    result_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=16,
                         verticalalignment='top', horizontalalignment='right',
                         fontweight='bold')

    ax.set_xlabel('Distance (m)', fontsize=12, color='white')
    ax.set_ylabel('Distance (m)', fontsize=12, color='white')
    ax.set_title('SATURATION ATTACK SIMULATION', fontsize=16,
                fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')

    # Calculate frame-to-time mapping
    frame_dt = sim.dt

    def get_time_for_frame(frame):
        return frame * frame_dt

    def get_status_at_time(t):
        # Count targets by status at time t
        incoming = engaged = destroyed = leaked = 0
        for target in sim.targets:
            # Find latest event before time t
            status = "incoming"
            for evt_time, evt_type, tid, iid in sim.state.events:
                if tid == target.target_id and evt_time <= t:
                    if evt_type == "launch":
                        status = "engaged"
                    elif evt_type == "hit":
                        status = "destroyed"
                    elif evt_type == "miss":
                        status = "incoming"
                    elif evt_type == "leak":
                        status = "leaked"

            if status == "incoming":
                incoming += 1
            elif status == "engaged":
                engaged += 1
            elif status == "destroyed":
                destroyed += 1
            elif status == "leaked":
                leaked += 1

        return incoming, engaged, destroyed, leaked

    def get_battery_status_at_time(t):
        launched = 0
        in_flight = 0
        for evt_time, evt_type, tid, iid in sim.state.events:
            if evt_time <= t:
                if evt_type == "launch":
                    launched += 1
                    in_flight += 1
                elif evt_type in ["hit", "miss"]:
                    in_flight -= 1
        remaining = sim.num_interceptors - launched
        return launched, in_flight, remaining

    def init():
        for poly in target_polys:
            poly.set_xy(create_cm_shape(shape_size))
        for poly in sam_polys:
            poly.set_xy(create_sam_shape(shape_size * 0.8))
            poly.set_alpha(0.0)
        for trail in target_trails + sam_trails:
            trail.set_data([], [])
        for marker in event_markers:
            marker.remove()
        event_markers.clear()
        threat_box.set_text('')
        battery_box.set_text('')
        time_box.set_text('')
        result_text.set_text('')
        return tuple(target_polys + sam_polys + target_trails + sam_trails +
                    [threat_box, battery_box, time_box, result_text])

    def animate(frame):
        current_time = get_time_for_frame(frame)

        # Update target positions and trails
        for i, target in enumerate(sim.targets):
            path = target.path_history
            if frame < len(path):
                pos = path[frame]
                heading = get_heading(path, frame)
                target_polys[i].set_xy(rotate_shape(create_cm_shape(shape_size), heading, pos))
                target_polys[i].set_alpha(1.0)
                target_trails[i].set_data([p[0] for p in path[:frame+1]],
                                          [p[1] for p in path[:frame+1]])
            else:
                # Target resolved - check final status
                if target.status == "destroyed":
                    target_polys[i].set_alpha(0.0)  # Hide destroyed
                elif target.status == "leaked":
                    target_polys[i].set_alpha(0.3)  # Dim leaked

        # Update interceptor positions and trails
        for j, interceptor in enumerate(sim.battery.interceptors):
            path = interceptor.path_history
            if len(path) > 0:
                # Find launch time for this interceptor
                launch_frame = None
                for evt_time, evt_type, tid, iid in sim.state.events:
                    if iid == j and evt_type == "launch":
                        launch_frame = int(evt_time / frame_dt)
                        break

                if launch_frame is not None and frame >= launch_frame:
                    path_idx = min(frame - launch_frame, len(path) - 1)
                    if path_idx >= 0 and path_idx < len(path):
                        pos = path[path_idx]
                        heading = get_heading(path, path_idx)
                        sam_polys[j].set_xy(rotate_shape(create_sam_shape(shape_size * 0.8),
                                                         heading, pos))
                        sam_polys[j].set_alpha(1.0 if interceptor.status == "in_flight" else 0.0)

                        if interceptor.status in ["hit", "miss", "expended"]:
                            sam_polys[j].set_alpha(0.0)

                        trail_end = min(path_idx + 1, len(path))
                        sam_trails[j].set_data([p[0] for p in path[:trail_end]],
                                               [p[1] for p in path[:trail_end]])
                else:
                    sam_polys[j].set_alpha(0.0)
            else:
                sam_polys[j].set_alpha(0.0)

        # Update info panels
        incoming, engaged, destroyed, leaked = get_status_at_time(current_time)
        launched, in_flight, remaining = get_battery_status_at_time(current_time)

        threat_str = "═══ THREATS ═══\n"
        threat_str += f"Incoming:  {incoming}\n"
        threat_str += f"Engaged:   {engaged}\n"
        threat_str += f"Destroyed: {destroyed}\n"
        threat_str += f"Leaked:    {leaked}"
        threat_box.set_text(threat_str)

        battery_str = "═══ BATTERY ═══\n"
        battery_str += f"Launched:  {launched}\n"
        battery_str += f"In-Flight: {in_flight}\n"
        battery_str += f"Remaining: {remaining}\n"
        reload_wait = sim.battery.time_until_ready(current_time)
        if reload_wait > 0 and remaining > 0:
            battery_str += f"Reload:    {reload_wait:.1f}s"
        else:
            battery_str += f"Status:    READY" if remaining > 0 else "Status:    EMPTY"
        battery_box.set_text(battery_str)

        time_box.set_text(f"Time: {current_time:.1f}s")

        # Final result
        if frame >= max_frames - 1:
            if sim.state.targets_destroyed == sim.num_targets:
                result_text.set_text("DEFENSE SUCCESS!")
                result_text.set_color('lime')
            elif sim.state.targets_leaked > 0:
                result_text.set_text(f"CITY HIT! ({sim.state.targets_leaked} leaked)")
                result_text.set_color('red')
            else:
                eff = sim.state.targets_destroyed / sim.num_targets * 100
                result_text.set_text(f"Defense: {eff:.0f}%")
                result_text.set_color('yellow')

        return tuple(target_polys + sam_polys + target_trails + sam_trails +
                    [threat_box, battery_box, time_box, result_text])

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=max_frames, interval=interval,
                         blit=True, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=30)
        print("Done!")

    plt.tight_layout()
    plt.show()
    return anim


def plot_saturation_attack_3d(sim: SaturationSimulation, save_path: str = None,
                               view_elev: float = 25, view_azim: float = -60):
    """
    3D visualization of a saturation attack scenario.

    Shows the full 3D flight paths of targets and interceptors with:
    - Earth surface (partial sphere)
    - Target trajectories (color-coded by status)
    - Interceptor trajectories (cyan)
    - Intercept points marked with explosions
    - City and SAM battery locations

    Args:
        sim: A completed SaturationSimulation
        save_path: Optional path to save the plot
        view_elev: Elevation angle for 3D view (degrees)
        view_azim: Azimuth angle for 3D view (degrees)
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get positions
    city_pos = sim.city_position
    battery_pos = sim.battery_position

    # Collect all positions for view bounds
    all_positions = []
    for target in sim.targets:
        all_positions.extend(target.path_history)
    for interceptor in sim.battery.interceptors:
        all_positions.extend(interceptor.path_history)
    all_positions.append(city_pos)
    all_positions.append(battery_pos)

    if len(all_positions) == 0:
        print("No data to plot!")
        return None

    all_positions = np.array(all_positions)

    # Convert to local tangent plane coordinates centered at city
    # This makes visualization cleaner than Earth-centered
    def to_local(pos):
        """Convert Earth-centered to local tangent plane at city."""
        # Local coordinate: X = East, Y = North, Z = Up
        # City is at the origin of local frame
        rel = pos - city_pos
        # City's "up" direction
        up = city_pos / np.linalg.norm(city_pos)
        # East is perpendicular to up and north (assume Y in global is roughly north)
        north = np.array([0, 1, 0])
        north = north - np.dot(north, up) * up
        north_norm = np.linalg.norm(north)
        if north_norm < 0.01:
            north = np.array([1, 0, 0])
            north = north - np.dot(north, up) * up
        north = north / np.linalg.norm(north)
        east = np.cross(north, up)

        # Project relative position to local coordinates
        return np.array([
            np.dot(rel, east),      # X: East
            np.dot(rel, north),     # Y: North
            np.dot(rel, up)         # Z: Up (altitude above surface)
        ])

    # Convert all positions to local frame
    city_local = to_local(city_pos)  # Should be [0, 0, 0]
    battery_local = to_local(battery_pos)

    # Collect all LOCAL positions for view bounds
    all_local_positions = [city_local, battery_local]

    # Color schemes
    target_colors = ['#ff4444', '#ff6644', '#ff8844', '#ffaa44', '#ffcc44', '#ffee44']
    sam_color = '#00ffff'

    # Plot target trajectories
    for i, target in enumerate(sim.targets):
        if len(target.path_history) < 2:
            continue

        path_local = np.array([to_local(p) for p in target.path_history])
        color = target_colors[i % len(target_colors)]

        # Status determines line style
        if target.status == "destroyed":
            linestyle = '-'
            alpha = 0.9
        elif target.status == "leaked":
            linestyle = '-'
            alpha = 0.9
        else:
            linestyle = '--'
            alpha = 0.6

        ax.plot3D(path_local[:, 0], path_local[:, 1], path_local[:, 2],
                  color=color, linestyle=linestyle, alpha=alpha, linewidth=2,
                  label=f'CM-{i+1} ({target.status})')
        all_local_positions.extend(path_local.tolist())

        # Mark start position
        ax.scatter([path_local[0, 0]], [path_local[0, 1]], [path_local[0, 2]],
                   color=color, s=50, marker='o')

        # Mark end position based on status
        if target.status == "destroyed":
            ax.scatter([path_local[-1, 0]], [path_local[-1, 1]], [path_local[-1, 2]],
                       color='orange', s=150, marker='*', edgecolors='yellow', linewidths=1)
        elif target.status == "leaked":
            ax.scatter([path_local[-1, 0]], [path_local[-1, 1]], [path_local[-1, 2]],
                       color='red', s=100, marker='x', linewidths=3)

    # Plot interceptor trajectories
    for i, interceptor in enumerate(sim.battery.interceptors):
        if len(interceptor.path_history) < 2:
            continue

        path_local = np.array([to_local(p) for p in interceptor.path_history])

        # Status determines line style
        if interceptor.status == "hit":
            linestyle = '-'
            alpha = 0.9
        elif interceptor.status == "miss":
            linestyle = ':'
            alpha = 0.6
        else:
            linestyle = '-'
            alpha = 0.7

        ax.plot3D(path_local[:, 0], path_local[:, 1], path_local[:, 2],
                  color=sam_color, linestyle=linestyle, alpha=alpha, linewidth=1.5,
                  label=f'SAM-{i} ({interceptor.status})' if i < 3 else None)
        all_local_positions.extend(path_local.tolist())

        # Mark launch point
        ax.scatter([path_local[0, 0]], [path_local[0, 1]], [path_local[0, 2]],
                   color='lime', s=30, marker='^')

    # Mark city
    ax.scatter([city_local[0]], [city_local[1]], [city_local[2]],
               color='yellow', s=300, marker='*', edgecolors='gold', linewidths=2,
               label='City', zorder=10)

    # Mark SAM battery
    ax.scatter([battery_local[0]], [battery_local[1]], [battery_local[2]],
               color='lime', s=150, marker='s', edgecolors='white', linewidths=1,
               label='SAM Battery', zorder=10)

    # Convert to numpy array for calculations
    all_local_positions = np.array(all_local_positions)

    # Draw ground plane (approximate)
    ground_size = max(
        abs(all_local_positions[:, 0]).max(),
        abs(all_local_positions[:, 1]).max()
    ) * 1.2
    ground_x = np.linspace(-ground_size, ground_size, 20)
    ground_y = np.linspace(-ground_size, ground_size, 20)
    ground_X, ground_Y = np.meshgrid(ground_x, ground_y)
    ground_Z = np.zeros_like(ground_X)
    ax.plot_surface(ground_X, ground_Y, ground_Z, alpha=0.1, color='darkgreen',
                    edgecolor='none', zorder=1)

    # Set labels and title
    ax.set_xlabel('East (m)', fontsize=11)
    ax.set_ylabel('North (m)', fontsize=11)
    ax.set_zlabel('Altitude (m)', fontsize=11)

    # Calculate defense effectiveness
    total_resolved = sim.state.targets_destroyed + sim.state.targets_leaked
    if total_resolved > 0:
        effectiveness = sim.state.targets_destroyed / total_resolved * 100
        title = f'3D SATURATION ATTACK - Defense: {effectiveness:.0f}%\n'
    else:
        title = '3D SATURATION ATTACK\n'
    title += f'Targets: {sim.num_targets} | Destroyed: {sim.state.targets_destroyed} | Leaked: {sim.state.targets_leaked}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Legend
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Make axes equal for proper 3D perspective
    max_range = max(
        abs(all_local_positions[:, 0]).max(),
        abs(all_local_positions[:, 1]).max()
    ) * 1.1
    max_alt = all_local_positions[:, 2].max() * 1.2

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max(max_alt, max_range * 0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved 3D plot to {save_path}")

    plt.show()
    return fig


def animate_saturation_attack_3d(sim: SaturationSimulation, interval: int = 50,
                                  save_path: str = None, view_elev: float = 25,
                                  view_azim: float = -60, rotate_view: bool = True):
    """
    Animated 3D visualization of a saturation attack scenario.

    Shows the simulation playing out in 3D with:
    - Animated target and interceptor trajectories
    - Color-coded status (incoming, engaged, destroyed, leaked)
    - Real-time status panels
    - Optional camera rotation for dramatic effect

    Args:
        sim: A completed SaturationSimulation
        interval: Milliseconds between frames
        save_path: Optional path to save animation (requires ffmpeg)
        view_elev: Initial elevation angle for 3D view (degrees)
        view_azim: Initial azimuth angle for 3D view (degrees)
        rotate_view: If True, slowly rotate the camera during animation
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    # Get positions
    city_pos = sim.city_position
    battery_pos = sim.battery_position

    # Collect all positions for view bounds
    all_positions = []
    for target in sim.targets:
        all_positions.extend(target.path_history)
    for interceptor in sim.battery.interceptors:
        all_positions.extend(interceptor.path_history)
    all_positions.append(city_pos)
    all_positions.append(battery_pos)

    if len(all_positions) == 0:
        print("No data to animate!")
        return None

    all_positions = np.array(all_positions)

    # Convert to local tangent plane coordinates centered at city
    def to_local(pos):
        """Convert Earth-centered to local tangent plane at city."""
        rel = pos - city_pos
        up = city_pos / np.linalg.norm(city_pos)
        north = np.array([0, 1, 0])
        north = north - np.dot(north, up) * up
        north_norm = np.linalg.norm(north)
        if north_norm < 0.01:
            north = np.array([1, 0, 0])
            north = north - np.dot(north, up) * up
        north = north / np.linalg.norm(north)
        east = np.cross(north, up)
        return np.array([
            np.dot(rel, east),
            np.dot(rel, north),
            np.dot(rel, up)
        ])

    # Precompute local paths for all entities
    target_paths_local = []
    for target in sim.targets:
        if target.path_history:
            path_local = np.array([to_local(p) for p in target.path_history])
            target_paths_local.append(path_local)
        else:
            target_paths_local.append(np.array([]))

    interceptor_paths_local = []
    for interceptor in sim.battery.interceptors:
        if interceptor.path_history:
            path_local = np.array([to_local(p) for p in interceptor.path_history])
            interceptor_paths_local.append(path_local)
        else:
            interceptor_paths_local.append(np.array([]))

    city_local = to_local(city_pos)
    battery_local = to_local(battery_pos)

    # Calculate view bounds from LOCAL coordinates (not Earth-centered)
    all_local_positions = []
    for path in target_paths_local:
        if len(path) > 0:
            all_local_positions.extend(path)
    for path in interceptor_paths_local:
        if len(path) > 0:
            all_local_positions.extend(path)
    all_local_positions.append(city_local)
    all_local_positions.append(battery_local)
    all_local_positions = np.array(all_local_positions)

    # Zoom in closer - focus on the intercept zone
    max_range = max(
        abs(all_local_positions[:, 0]).max(),
        abs(all_local_positions[:, 1]).max(),
        all_local_positions[:, 2].max()  # Z is altitude, always positive
    ) * 0.5  # Reduced from 1.1 to zoom in

    # Color schemes
    target_colors = ['#ff4444', '#ff6644', '#ff8844', '#ffaa44', '#ffcc44', '#ffee44']
    sam_color = '#00ffff'

    # Create line objects for trails (larger for zoomed view)
    target_trails = []
    target_markers = []
    for i in range(len(sim.targets)):
        color = target_colors[i % len(target_colors)]
        trail, = ax.plot3D([], [], [], color=color, linewidth=3, alpha=0.8)
        target_trails.append(trail)
        marker, = ax.plot3D([], [], [], 'o', color=color, markersize=12)
        target_markers.append(marker)

    interceptor_trails = []
    interceptor_markers = []
    for i in range(len(sim.battery.interceptors)):
        trail, = ax.plot3D([], [], [], color=sam_color, linewidth=2.5, alpha=0.9)
        interceptor_trails.append(trail)
        marker, = ax.plot3D([], [], [], '^', color='lime', markersize=10)
        interceptor_markers.append(marker)

    # Hit markers (explosions)
    hit_markers = []

    # Static elements (larger for zoomed view)
    ax.scatter([city_local[0]], [city_local[1]], [city_local[2]],
               color='yellow', s=600, marker='*', edgecolors='gold', linewidths=3,
               label='City', zorder=10)

    ax.scatter([battery_local[0]], [battery_local[1]], [battery_local[2]],
               color='lime', s=350, marker='s', edgecolors='white', linewidths=2,
               label='SAM Battery', zorder=10)

    # Ground plane
    ground_size = max_range * 1.2
    ground_x = np.linspace(-ground_size, ground_size, 10)
    ground_y = np.linspace(-ground_size, ground_size, 10)
    ground_X, ground_Y = np.meshgrid(ground_x, ground_y)
    ground_Z = np.zeros_like(ground_X)
    ax.plot_surface(ground_X, ground_Y, ground_Z, alpha=0.1, color='darkgreen',
                    edgecolor='none', zorder=1)

    # Set axis properties
    max_alt = all_local_positions[:, 2].max() * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max(max_alt, max_range * 0.3))
    ax.set_xlabel('East (m)', fontsize=10, color='white')
    ax.set_ylabel('North (m)', fontsize=10, color='white')
    ax.set_zlabel('Altitude (m)', fontsize=10, color='white')
    ax.tick_params(colors='white')

    # Title and info boxes
    title_text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, fontsize=14,
                           fontweight='bold', color='white', ha='center')

    status_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                            color='white', family='monospace', va='top',
                            bbox=dict(boxstyle='round', facecolor='#1a0a0a',
                                     alpha=0.9, edgecolor='red'))

    battery_text = ax.text2D(0.02, 0.70, '', transform=ax.transAxes, fontsize=11,
                             color='white', family='monospace', va='top',
                             bbox=dict(boxstyle='round', facecolor='#0a1a0a',
                                      alpha=0.9, edgecolor='lime'))

    result_text = ax.text2D(0.98, 0.95, '', transform=ax.transAxes, fontsize=14,
                            fontweight='bold', ha='right', va='top')

    # Find max frames
    max_frames = max(
        max((len(p) for p in target_paths_local if len(p) > 0), default=1),
        max((len(p) for p in interceptor_paths_local if len(p) > 0), default=1)
    )

    # Frame time mapping
    frame_dt = sim.dt

    def get_status_at_frame(frame):
        """Get target/battery status at given frame."""
        t = frame * frame_dt
        incoming = engaged = destroyed = leaked = 0
        in_flight = 0

        for target in sim.targets:
            status = "incoming"
            for evt_time, evt_type, tid, iid in sim.state.events:
                if tid == target.target_id and evt_time <= t:
                    if evt_type == "launch":
                        status = "engaged"
                    elif evt_type == "hit":
                        status = "destroyed"
                    elif evt_type == "miss":
                        status = "incoming"
                    elif evt_type == "leak":
                        status = "leaked"

            if status == "incoming":
                incoming += 1
            elif status == "engaged":
                engaged += 1
            elif status == "destroyed":
                destroyed += 1
            elif status == "leaked":
                leaked += 1

        for interceptor in sim.battery.interceptors:
            if interceptor.launch_time and interceptor.launch_time <= t:
                # Check if still in flight
                is_hit = any(evt_type == "hit" and iid == interceptor.interceptor_id
                            and evt_time <= t
                            for evt_time, evt_type, tid, iid in sim.state.events)
                is_miss = any(evt_type == "miss" and iid == interceptor.interceptor_id
                             and evt_time <= t
                             for evt_time, evt_type, tid, iid in sim.state.events)
                if not is_hit and not is_miss:
                    in_flight += 1

        return incoming, engaged, destroyed, leaked, in_flight

    def init():
        """Initialize animation."""
        for trail in target_trails:
            trail.set_data_3d([], [], [])
        for marker in target_markers:
            marker.set_data_3d([], [], [])
        for trail in interceptor_trails:
            trail.set_data_3d([], [], [])
        for marker in interceptor_markers:
            marker.set_data_3d([], [], [])
        title_text.set_text('')
        status_text.set_text('')
        battery_text.set_text('')
        result_text.set_text('')
        return (target_trails + target_markers + interceptor_trails +
                interceptor_markers + [title_text, status_text, battery_text, result_text])

    def animate(frame):
        """Update animation frame."""
        nonlocal hit_markers

        t = frame * frame_dt

        # Update target trails and markers
        for i, (trail, marker, path) in enumerate(zip(target_trails, target_markers,
                                                       target_paths_local)):
            if len(path) > 0 and frame < len(path):
                # Trail up to current frame
                trail.set_data_3d(path[:frame+1, 0], path[:frame+1, 1], path[:frame+1, 2])
                # Current position marker
                marker.set_data_3d([path[frame, 0]], [path[frame, 1]], [path[frame, 2]])
            elif len(path) > 0 and frame >= len(path):
                # Show full trail, hide marker (destroyed or leaked)
                trail.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                marker.set_data_3d([], [], [])

        # Update interceptor trails and markers
        for i, (trail, marker, path, interceptor) in enumerate(
                zip(interceptor_trails, interceptor_markers,
                    interceptor_paths_local, sim.battery.interceptors)):
            if len(path) > 0:
                # Find the frame offset based on launch time
                if interceptor.launch_time:
                    launch_frame = int(interceptor.launch_time / frame_dt)
                    relative_frame = frame - launch_frame
                    if relative_frame >= 0 and relative_frame < len(path):
                        trail.set_data_3d(path[:relative_frame+1, 0],
                                         path[:relative_frame+1, 1],
                                         path[:relative_frame+1, 2])
                        marker.set_data_3d([path[relative_frame, 0]],
                                          [path[relative_frame, 1]],
                                          [path[relative_frame, 2]])
                    elif relative_frame >= len(path):
                        trail.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                        marker.set_data_3d([], [], [])
                    else:
                        trail.set_data_3d([], [], [])
                        marker.set_data_3d([], [], [])

        # Check for hit events at this frame and add explosion markers
        for evt_time, evt_type, tid, iid in sim.state.events:
            if evt_type == "hit" and abs(evt_time - t) < frame_dt * 2:
                # Find the intercept position
                target_path = target_paths_local[tid]
                if len(target_path) > 0:
                    evt_frame = min(int(evt_time / frame_dt), len(target_path) - 1)
                    pos = target_path[evt_frame]
                    # Add explosion marker (larger for zoomed view)
                    explosion = ax.scatter([pos[0]], [pos[1]], [pos[2]],
                                          color='orange', s=500, marker='*',
                                          edgecolors='yellow', linewidths=3, alpha=0.95)
                    hit_markers.append((explosion, frame))

        # Fade out old hit markers
        new_hit_markers = []
        for explosion, hit_frame in hit_markers:
            age = frame - hit_frame
            if age < 30:  # Keep visible for 30 frames
                alpha = max(0, 1.0 - age / 30.0)
                explosion.set_alpha(alpha)
                new_hit_markers.append((explosion, hit_frame))
            else:
                explosion.remove()
        hit_markers = new_hit_markers

        # Rotate view if enabled
        if rotate_view:
            new_azim = view_azim + frame * 0.2
            ax.view_init(elev=view_elev, azim=new_azim)
        else:
            ax.view_init(elev=view_elev, azim=view_azim)

        # Update title
        title_text.set_text(f'3D SATURATION ATTACK  |  Time: {t:.1f}s')

        # Update status panels
        incoming, engaged, destroyed, leaked, in_flight = get_status_at_frame(frame)

        status_text.set_text(
            f'THREATS\n'
            f'─────────\n'
            f'Incoming:  {incoming}\n'
            f'Engaged:   {engaged}\n'
            f'Destroyed: {destroyed}\n'
            f'Leaked:    {leaked}'
        )

        launched = sim.state.interceptors_launched
        remaining = sim.num_interceptors - launched
        battery_text.set_text(
            f'BATTERY\n'
            f'─────────\n'
            f'In-Flight: {in_flight}\n'
            f'Launched:  {launched}\n'
            f'Remaining: {remaining}'
        )

        # Final result
        if frame >= max_frames - 10:
            if sim.state.targets_leaked > 0:
                result_text.set_text(f"CITY HIT! ({sim.state.targets_leaked} leaked)")
                result_text.set_color('red')
            else:
                eff = sim.state.targets_destroyed / sim.num_targets * 100
                result_text.set_text(f"Defense: {eff:.0f}%")
                result_text.set_color('lime')

        return (target_trails + target_markers + interceptor_trails +
                interceptor_markers + [title_text, status_text, battery_text, result_text])

    # Create animation (blit=False for 3D)
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=max_frames, interval=interval,
                         blit=False, repeat=True)

    if save_path:
        print(f"Saving 3D animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=30,
                  savefig_kwargs={'facecolor': '#1a1a2e'})
        print("Done!")

    plt.tight_layout()
    plt.show()
    return anim


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_engagement(pursuer: Aircraft, target: Aircraft, analysis: dict,
                    time_range: float = None, save_path: str = None):
    """
    Visualize the engagement geometry.
    
    Shows:
    - Current positions and velocity vectors
    - Projected paths
    - Intercept point (if possible)
    - Lead angle illustration
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Determine plot bounds
    if time_range is None:
        time_range = analysis["intercept_time"] * 1.5 if analysis["intercept_time"] else 60
    
    # Plot current positions
    ax.plot(*pursuer.position, 'bo', markersize=15, label=f'Pursuer ({pursuer.name})')
    ax.plot(*target.position, 'ro', markersize=15, label=f'Target ({target.name})')
    
    # Plot velocity vectors (scaled for visibility)
    scale = time_range * 0.15  # Scale vectors to be visible
    ax.arrow(pursuer.position[0], pursuer.position[1],
             pursuer.velocity[0] * scale / pursuer.speed if pursuer.speed > 0 else 0,
             pursuer.velocity[1] * scale / pursuer.speed if pursuer.speed > 0 else 0,
             head_width=500, head_length=300, fc='blue', ec='blue', alpha=0.7)
    ax.arrow(target.position[0], target.position[1],
             target.velocity[0] * scale / target.speed if target.speed > 0 else 0,
             target.velocity[1] * scale / target.speed if target.speed > 0 else 0,
             head_width=500, head_length=300, fc='red', ec='red', alpha=0.7)
    
    # Plot projected paths (constant velocity)
    times = np.linspace(0, time_range, 100)
    pursuer_path = np.array([pursuer.position_at_time(t) for t in times])
    target_path = np.array([target.position_at_time(t) for t in times])
    
    ax.plot(pursuer_path[:, 0], pursuer_path[:, 1], 'b--', alpha=0.5, 
            label='Pursuer path (no course change)')
    ax.plot(target_path[:, 0], target_path[:, 1], 'r--', alpha=0.5,
            label='Target projected path')
    
    # If intercept is possible, show it
    if analysis["intercept_possible"]:
        intercept_pt = analysis["intercept_point"]
        ax.plot(*intercept_pt, 'g*', markersize=20, label='Intercept point')
        
        # Draw intercept course (the path pursuer SHOULD take)
        intercept_heading_rad = np.radians(analysis["required_heading"])
        intercept_vel = pursuer.max_speed * np.array([np.cos(intercept_heading_rad),
                                                       np.sin(intercept_heading_rad)])
        intercept_path = np.array([pursuer.position + intercept_vel * t 
                                   for t in np.linspace(0, analysis["intercept_time"], 50)])
        ax.plot(intercept_path[:, 0], intercept_path[:, 1], 'g-', linewidth=2,
                label='Intercept course')
        
        # Annotate intercept time
        ax.annotate(f't = {analysis["intercept_time"]:.1f}s',
                    xy=intercept_pt, xytext=(intercept_pt[0] + 1000, intercept_pt[1] + 1000),
                    fontsize=12, color='green',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Labels and formatting
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title('Intercept Geometry Analysis', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add analysis text box
    textstr = '\n'.join([
        f"Initial Range: {analysis['initial_range']:.0f} m",
        f"Closure Rate: {analysis['closure_rate']:.1f} m/s",
        f"Intercept Possible: {analysis['intercept_possible']}",
    ])
    if analysis["intercept_possible"]:
        textstr += f"\nIntercept Time: {analysis['intercept_time']:.2f} s"
        textstr += f"\nLead Angle: {analysis['lead_angle']:.1f}°"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()
    return fig


# =============================================================================
# EXAMPLE SCENARIOS
# =============================================================================

def run_scenario(name: str, pursuer: Aircraft, target: Aircraft):
    """Run and visualize a single scenario."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print('='*60)
    print(f"\nPursuer: {pursuer}")
    print(f"Target:  {target}")
    
    analysis = analyze_engagement(pursuer, target)
    
    print(f"\n--- ANALYSIS ---")
    print(f"Initial Range: {analysis['initial_range']:.0f} m ({analysis['initial_range']/1000:.1f} km)")
    print(f"Closure Rate: {analysis['closure_rate']:.1f} m/s")
    
    if analysis["intercept_possible"]:
        print(f"\n✓ INTERCEPT POSSIBLE")
        print(f"  Time to intercept: {analysis['intercept_time']:.2f} seconds")
        print(f"  Intercept point: ({analysis['intercept_point'][0]:.0f}, {analysis['intercept_point'][1]:.0f})")
        print(f"  Required heading: {analysis['required_heading']:.1f}°")
        print(f"  Lead angle (turn): {analysis['lead_angle']:.1f}°")
    else:
        print(f"\n✗ INTERCEPT NOT POSSIBLE")
        print(f"  Target is too fast or geometry unfavorable")
    
    return analysis


def main():
    """Run a single intercept scenario demonstration."""

    print("\n" + "="*60)
    print("GLOBAL MISSILE DEFENSE SIMULATION")
    print("With Realistic Flight Dynamics")
    print("="*60)

    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ==========================================================================
    # GLOBAL SCALE SCENARIO: Cruise missile attack on city, SAM defense
    # Using Earth curvature - Earth radius ~6,371 km
    # Coordinates: x = horizontal distance along surface, y = altitude from center
    # ==========================================================================

    EARTH_RADIUS = 6_371_000  # meters (6,371 km)

    # City location (what we're defending) - on Earth's surface
    # We'll place the "scene" so the city is at angle 0 (right side of view)
    city_angle = 0  # radians from center of view
    city_x = EARTH_RADIUS * np.sin(city_angle)
    city_y = EARTH_RADIUS * np.cos(city_angle)

    # SAM site is near the city (50km away along surface)
    sam_surface_distance = 50_000  # 50km from city along Earth's surface
    sam_angle = sam_surface_distance / EARTH_RADIUS  # convert to radians
    sam_x = EARTH_RADIUS * np.sin(-sam_angle)  # west of city
    sam_y = EARTH_RADIUS * np.cos(-sam_angle)

    # Incoming BrahMos-II class cruise missile
    # Starting 120km from city = ~70km from SAM site
    missile_surface_distance = 120_000  # 120km from city
    missile_altitude = 15_000  # 15km altitude (typical cruise altitude for supersonic CM)
    missile_angle = missile_surface_distance / EARTH_RADIUS
    missile_radius = EARTH_RADIUS + missile_altitude
    missile_x = missile_radius * np.sin(-missile_angle)  # coming from the west
    missile_y = missile_radius * np.cos(-missile_angle)

    # Cruise missile velocity - hypersonic at ~Mach 3, heading toward city
    cruise_speed = 1000.0  # m/s (~Mach 3)
    # Direction from missile to city
    to_city = np.array([city_x - missile_x, city_y - missile_y])
    to_city_unit = to_city / np.linalg.norm(to_city)
    cruise_missile_vx = cruise_speed * to_city_unit[0]
    cruise_missile_vy = cruise_speed * to_city_unit[1]

    print(f"\nEarth Radius: {EARTH_RADIUS/1000:.0f} km")
    print(f"City Position: surface (reference point)")
    print(f"SAM Site: {sam_surface_distance/1000:.0f} km from city")
    print(f"Incoming Missile: {missile_surface_distance/1000:.0f} km away, {missile_altitude/1000:.0f} km altitude")

    # Defender: Patriot PAC-3 MSE (Missile Segment Enhancement)
    # Real specs: Mach 5, range 35km, altitude 24km, dual-pulse motor
    # Specifically designed for cruise missile and tactical ballistic missile defense
    pursuer = Aircraft(
        name="PAC-3 MSE",
        position=np.array([sam_x, sam_y]),
        velocity=np.array([150.0, 600.0]),     # Initial launch velocity
        max_speed=1700.0,                       # 1.7 km/s (Mach 5) - actual PAC-3 spec
        max_acceleration=200.0,                 # High thrust-to-weight ratio
        max_g=60.0,                             # 60g - 180 attitude control thrusters
        burn_time=45.0,                         # Dual-pulse solid motor
        drag_coefficient=0.0001,                # Low drag design
        is_missile=True                         # Enable gravity with proper compensation
    )

    # Attacker: BrahMos-II class hypersonic cruise missile
    # Specs based on BrahMos: Mach 2.8-3, 15km cruise altitude, 450km range
    target = Aircraft(
        name="BrahMos-II",
        position=np.array([missile_x, missile_y]),
        velocity=np.array([cruise_missile_vx, cruise_missile_vy]),
        max_speed=1000.0,                       # ~Mach 2.9 cruise speed
        max_acceleration=15.0,                  # Ramjet sustainer
        max_g=5.0,                              # Limited maneuverability at Mach 3
        burn_time=300.0,                        # Ramjet duration
        drag_coefficient=0.0,                   # Thrust compensates drag
        is_missile=False                        # Maintains altitude
    )

    # Analyze the engagement
    analysis = run_scenario("City Defense - Cruise Missile Intercept", pursuer, target)

    # Generate visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION...")
    print("="*60)

    filename = os.path.join(output_dir, "scenario.png")
    plot_engagement(pursuer, target, analysis, save_path=filename)
    print(f"  Saved: {filename}")

    # Run real-time simulation
    print("\n" + "="*60)
    print("RUNNING SIMULATION...")
    print("="*60)

    # Radar system parameters
    RADAR_RANGE = 250_000          # 250km detection range (early warning)
    LAUNCH_DELAY = 5.0             # 5 seconds from detection to launch

    print(f"Radar Range: {RADAR_RANGE/1000:.0f} km")
    print(f"Launch Delay: {LAUNCH_DELAY:.0f} seconds")

    sim = InterceptSimulation(
        pursuer=pursuer,
        target=target,
        dt=0.05,                       # 50ms time steps (fine resolution for intercept)
        max_time=900.0,                # 15 minutes max (longer for radar wait)
        intercept_range=200.0,         # 200m kill radius
        pursuer_turn_rate=25.0,        # Base turn rate (modified by g-limits)
        target_turn_rate=2.8,          # ~5g at Mach 3 = 2.8°/s max turn
        target_maneuver_intensity=0.1, # Cruise missiles mostly fly straight
        evasion_radius=0.0,            # No evasion - BrahMos has no missile warning system
        radar_range=RADAR_RANGE,       # Radar detection range
        launch_delay=LAUNCH_DELAY,     # Time from detection to launch
        earth_radius=EARTH_RADIUS,     # For radar horizon calculation
        enable_physics=True            # Enable realistic flight dynamics!
    )

    # Store Earth radius for visualization
    sim.earth_radius = EARTH_RADIUS
    sim.city_position = np.array([city_x, city_y])

    state = sim.run(verbose=True)
    plot_simulation_result(sim, save_path=os.path.join(output_dir, "sim_result.png"))

    # Animate with Earth curvature
    print("\n>>> Starting animation...")
    print("Close the window to exit.")
    animate_simulation_earth(sim, interval=20)

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


def run_saturation_demo():
    """
    Run a saturation attack demonstration.

    Simulates 6 incoming cruise missiles against a SAM battery with 8 interceptors.
    Uses 85% Pk (probability of kill) per intercept attempt.
    Shows 3D animated visualization.
    """
    print("\n" + "="*70)
    print("       SATURATION ATTACK DEMONSTRATION (3D)")
    print("       6 Cruise Missiles vs 8 SAM Interceptors")
    print("="*70)

    EARTH_RADIUS = 6_371_000  # meters

    # City position (3D: Earth-centered coordinates)
    city_pos = np.array([0.0, 0.0, EARTH_RADIUS])

    # SAM battery position (50km from city, 3D)
    sam_angle = 50_000 / EARTH_RADIUS
    battery_pos = np.array([
        -EARTH_RADIUS * np.sin(sam_angle),
        0.0,
        EARTH_RADIUS * np.cos(sam_angle)
    ])

    # Create and run saturation simulation
    sim = SaturationSimulation(
        num_targets=6,            # 6 incoming cruise missiles
        num_interceptors=8,       # 8 SAM interceptors
        pk_single=0.85,           # 85% probability of kill
        reload_time=5.0,          # 5 seconds between launches
        max_simultaneous=6,       # Up to 6 SAMs in flight at once
        city_position=city_pos,
        battery_position=battery_pos,
        earth_radius=EARTH_RADIUS,
        dt=0.05,                  # 50ms time steps (finer resolution)
        max_time=300.0,           # 5 minute max
        intercept_range=2000.0    # 2km kill radius (large fragmentation warhead)
    )

    # Run the simulation
    state = sim.run(verbose=True)

    # Show results summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Cruise Missiles:  {sim.num_targets}")
    print(f"  SAM Interceptors: {sim.num_interceptors} (Pk={sim.battery.pk_single:.0%})")
    print(f"  ─────────────────────────────────")
    print(f"  Destroyed:        {state.targets_destroyed}")
    print(f"  Leaked (City Hit):{state.targets_leaked}")
    print(f"  SAMs Used:        {state.interceptors_launched}")
    print(f"  SAMs Remaining:   {sim.num_interceptors - state.interceptors_launched}")

    if sim.num_targets > 0:
        effectiveness = state.targets_destroyed / sim.num_targets * 100
        print(f"  ─────────────────────────────────")
        if state.targets_leaked == 0:
            print(f"  Result: CITY DEFENDED! ({effectiveness:.0f}% effective)")
        else:
            print(f"  Result: CITY HIT! {state.targets_leaked} missile(s) got through")
            print(f"  Defense Effectiveness: {effectiveness:.0f}%")

    # Visualize the engagement in 3D
    print("\n>>> Starting 3D animation...")
    print("Close the window to exit.")
    animate_saturation_attack_3d(sim, interval=30, rotate_view=False)

    print("\n" + "="*70)
    print("SATURATION ATTACK DEMO COMPLETE")
    print("="*70)

    return sim, state


if __name__ == "__main__":
    # Run saturation attack demo (new feature)
    run_saturation_demo()

    # To run the original single-target demo, uncomment:
    # main()