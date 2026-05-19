# tvc-sim

Model rocket simulation sandbox.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
python -m tvc_sim
```

## UI

```sh
streamlit run src/tvc_sim_ui/streamlit_app.py
```

The legacy entry point still works:

```sh
python sim.py
```

## Structure

```text
src/tvc_sim/motor.py     thrust curve loading and interpolation
src/tvc_sim/rocket.py    rocket state, dynamics, and integration
src/tvc_sim/sim.py       default simulation setup
src/tvc_sim/plotting.py  Matplotlib output
src/tvc_sim_ui/          Streamlit UI
```

`Rocket` separates physical parameters, dynamic state, and control inputs:

```text
RocketParams   mass, moment arm, pitch moment of inertia
RocketState    time, position, velocity, pitch, acceleration
RocketInputs   thrust and gimbal angle
```
