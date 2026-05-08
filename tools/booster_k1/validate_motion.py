import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.append("mimickit")

import anim.mjcf_char_model as mjcf_char_model
import anim.motion as motion
import anim.motion_lib as motion_lib


K1_CHAR_FILE = "data/assets/booster_k1/k1.xml"
K1_FRAME_SIZE = 28
K1_DOF_SIZE = 22
K1_BODY_COUNT = 25
K1_KEY_BODIES = [
    "left_foot_link",
    "right_foot_link",
    "Head_2",
    "left_hand_end_ball",
    "right_hand_end_ball",
]


def _load_k1_model(device: str):
    char_model = mjcf_char_model.MJCFCharModel(device)
    char_model.load(K1_CHAR_FILE)

    body_names = char_model.get_body_names()
    if len(body_names) != K1_BODY_COUNT:
        raise ValueError(
            f"Expected {K1_BODY_COUNT} K1 bodies, found {len(body_names)}."
        )

    dof_size = char_model.get_dof_size()
    if dof_size != K1_DOF_SIZE:
        raise ValueError(f"Expected {K1_DOF_SIZE} K1 DoFs, found {dof_size}.")

    for body_name in K1_KEY_BODIES:
        char_model.get_body_id(body_name)

    return char_model


def _resolve_motion_files(path: Path) -> list[Path]:
    if path.suffix != ".yaml":
        return [path]

    with path.open("r") as stream:
        config = yaml.safe_load(stream)

    motion_entries = config.get("motions", [])
    if not motion_entries:
        raise ValueError(f"Dataset has no motions: {path}")

    motion_files = []
    for entry in motion_entries:
        motion_file = Path(entry["file"])
        motion_files.append(motion_file)

    return motion_files


def _validate_motion_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing motion file: {path}")

    motion_data = motion.load_motion(str(path))
    frame_shape = motion_data.frames.shape
    if len(frame_shape) != 2 or frame_shape[1] != K1_FRAME_SIZE:
        raise ValueError(
            f"{path} has frame shape {frame_shape}; expected (*, {K1_FRAME_SIZE})."
        )

    if motion_data.fps <= 0:
        raise ValueError(f"{path} has non-positive fps: {motion_data.fps}")

    return motion_data


def validate(path: Path, device: str):
    if not Path(K1_CHAR_FILE).is_file():
        raise FileNotFoundError(f"Missing K1 character file: {K1_CHAR_FILE}")

    char_model = _load_k1_model(device)

    motion_files = _resolve_motion_files(path)
    for motion_file in motion_files:
        _validate_motion_file(motion_file)

    lib = motion_lib.MotionLib(
        motion_file=str(path),
        kin_char_model=char_model,
        device=device,
    )
    motion_ids = lib.sample_motions(1)
    motion_times = torch.zeros_like(motion_ids, dtype=torch.float32, device=device)
    root_pos, root_rot, _, _, joint_rot, _ = lib.calc_motion_frame(
        motion_ids, motion_times
    )
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    print(f"Validated K1 motion input: {path}")
    print(f"  motion files: {len(motion_files)}")
    print(f"  dofs: {char_model.get_dof_size()}")
    print(f"  bodies: {len(char_model.get_body_names())}")
    print(f"  sampled body_pos shape: {tuple(body_pos.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Booster K1 MimicKit motion files or dataset YAMLs."
    )
    parser.add_argument("motion_file", help="Path to a MimicKit .pkl or dataset .yaml")
    parser.add_argument("--device", default="cpu", help="Torch device to validate on")
    args = parser.parse_args()

    validate(Path(args.motion_file), args.device)


if __name__ == "__main__":
    main()
