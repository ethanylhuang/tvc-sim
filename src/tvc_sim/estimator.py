from dataclasses import dataclass


@dataclass(frozen=True)
class EstimatedState:
    x: float = 0.0
    y: float = 0.0
    x_vel: float = 0.0
    y_vel: float = 0.0
    pitch: float = 0.0
    pitch_vel: float = 0.0


class ImuStateEstimator:
    def __init__(self):
        self.reset()

    def reset(self, initial_state=None):
        self.x = 0.0
        self.y = 0.0
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.pitch = 0.0
        self.pitch_vel = 0.0
        self.last_time = None
        self.initialized = False

        if initial_state is None:
            return

        self.x = initial_state.x
        self.y = initial_state.y
        self.x_vel = initial_state.x_vel
        self.y_vel = initial_state.y_vel
        self.pitch = initial_state.pitch
        self.pitch_vel = initial_state.pitch_vel
        self.last_time = initial_state.time
        self.initialized = True

    def update(self, readings, dt):
        if not self.initialized:
            self.pitch_vel = readings.pitch_vel
            self.last_time = readings.time
            self.initialized = True
            return self.estimated_state

        elapsed = dt
        if self.last_time is not None:
            elapsed = max(0.0, readings.time - self.last_time)

        x_vel = self.x_vel + readings.x_accel * elapsed
        y_vel = self.y_vel + readings.y_accel * elapsed
        pitch_vel = readings.pitch_vel

        x = self.x + x_vel * elapsed
        y = self.y + y_vel * elapsed
        pitch = self.pitch + pitch_vel * elapsed

        if y <= 0.0:
            x = self.x
            y = 0.0
            if self.y <= 0.0:
                pitch = self.pitch

        self.x = x
        self.y = y
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.pitch = pitch
        self.pitch_vel = pitch_vel
        self.last_time = readings.time
        return self.estimated_state

    @property
    def estimated_state(self):
        return EstimatedState(
            x=self.x,
            y=self.y,
            x_vel=self.x_vel,
            y_vel=self.y_vel,
            pitch=self.pitch,
            pitch_vel=self.pitch_vel,
        )
