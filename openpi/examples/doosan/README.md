# Doosan Runtime Example

This example explains how to reuse the released $\pi_{0.5}$-DROID checkpoint on a
Doosan (UR5e-class) arm without retraining. The only requirement is to pad the
6‑DoF(+gripper) state/actions to the 32‑dimensional format expected by the
checkpoint.

## 1. Start the policy server

```bash
cd ~/openpi
uv run scripts/serve_policy.py --env=DOOSAN --port=8000 --default_prompt="청소해 줘"
```

This loads:

- Config `pi05_doosan_runtime` (same $\pi_{0.5}$ model, Doosan-specific transforms)
- Checkpoint `gs://openpi-assets/checkpoints/pi05_droid`
- Input pipeline: `DoosanInputs` ⟶ normalization ⟶ model ⟶ `DoosanOutputs`

## 2. ROS bridge (example)

Run the bridge on the robot computer to gather camera/joint topics and send them
to the policy server via the websocket client:

```bash
uv run python examples/doosan/ros_bridge.py \
  --policy-host 10.0.0.5 --policy-port 8000 \
  --prompt "청소해 줘"
```

Expected ROS topics:

| Topic                               | Description            |
|------------------------------------|------------------------|
| `/camera/top/color/image_raw`      | Top view (RGB)         |
| `/camera/wrist/color/image_raw`    | Wrist camera RGB       |
| `/camera/front/color/image_raw`    | Front view RGB         |
| `/dsr01/joint_states`              | 6 joint positions      |

The bridge converts these into the dictionary format consumed by
`DoosanInputs`, automatically pads the state/action vectors to 32 dimensions,
and sends observation dicts to the websocket policy server. The resulting
`actions` array (first 7 dims) can be streamed back to the robot controller.
