"""
custom_pickle_dataset.py

PyTorch Dataset to train OpenVLA directly from a directory of pickle trajectories.

Assumptions per trajectory file (dict):
  - data: List[step], each step has
      - observation: {"image": uint8 HWC, "prompt": str}
      - env_action: {"world_vector": (3,), "rot_axangle": (3,), "gripper": (1,), ...}
  - task_description: str (fallback if per-step prompt is missing)

Each sample returns a dict with keys compatible with the finetune loop:
  - pixel_values: torch tensor from provided image_transform
  - input_ids: torch.LongTensor
  - labels: torch.LongTensor (masked to only supervise action tokens)

Also exposes `dataset_statistics` with action quantiles q01 and q99 for de-normalization at inference.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from rotation_utils import axangle2euler

class CustomPickleDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer,
        image_transform: Callable[[Image.Image], torch.Tensor],
        prompt_builder_fn: type[PromptBuilder],
    ) -> None:
        self.root_dir = root_dir
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        # Discover files
        self.files: List[str] = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        assert len(self.files) > 0, f"No .pkl files found in {root_dir}"
        self.index: List[dict[str, int]] = []

        self.action_norm_stats = {'action': {'mask': [True, True, True, True, True, True, False], 'max': [2.9984593391418457, 22.09052848815918, 2.7507524490356445, 1.570636510848999, 1.5321086645126343, 1.5691522359848022, 1.0], 'mean': [0.006987582892179489, 0.006265917327255011, -0.01262515690177679, 0.04333311319351196, -0.005756212864071131, 0.0009130256366916001, 0.5354204773902893], 'min': [-2.0204520225524902, -5.497899532318115, -2.031663417816162, -1.569917917251587, -1.569892168045044, -1.570419430732727, 0.0], 'q01': [-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704, 0.0], 'q99': [0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094, 1.0], 'std': [0.0692116990685463, 0.05970962345600128, 0.07353084534406662, 0.15610496699810028, 0.13164450228214264, 0.14593800902366638, 0.497110515832901]}, 'num_trajectories': 87212, 'num_transitions': 3786400, 'proprio': {'max': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'mean': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'min': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'q01': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'q99': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}

        # ---- your loop, modified to include inversion of gripper ----
        for fidx, fpath in enumerate(self.files):
            with open(fpath, "rb") as fh:
                traj = pickle.load(fh)
        
            if traj.get('success', True) is False:
                continue

            steps: Sequence[dict] = traj.get("data", [])
            if not steps:
                continue

            # Config (use dataset attrs if present)
            initial_prev_open = 0
            sticky_repeats = 15
            validate_sticky = True

            # Precompute absolute open_gripper across the whole trajectory
            open_abs_seq = self._compute_open_gripper_sequence(
                steps,
                initial_prev_open=initial_prev_open,
                sticky_gripper_num_repeat=sticky_repeats,
                validate=validate_sticky,
            )

            for sidx, step in enumerate(steps):
                self.index.append({"file_idx": fidx, "step_idx": sidx, "open_abs_seq": open_abs_seq[sidx]})


        # Expose for save_dataset_statistics()
        self.dataset_statistics = {
            "custom_pickle_dataset": {
                "action": {"q01": 0, "q99": 1},
            }
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        item = self.index[idx]
        fpath = self.files[item["file_idx"]]
        with open(fpath, "rb") as fh:
            traj = pickle.load(fh)

        step = traj["data"][item["step_idx"]]

        # Image
        image_arr = step["observation"]["image"]  # uint8 HWC
        image = Image.fromarray(image_arr)
        pixel_values = self.image_transform(image)

        # Instruction
        instruction = step["observation"].get("prompt")
        if instruction is None or len(str(instruction)) == 0:
            instruction = traj.get("task_description", "")

        # Action (7D)
        env_action = step["env_action"]
        world = np.asarray(env_action["world_vector"], dtype=np.float32).reshape(-1)  # (3,)
        rot = np.asarray(env_action["rot_axangle"], dtype=np.float32).reshape(-1)    # (3,)
        grip_abs = np.asarray([self.index[idx]["open_abs_seq"]], dtype=np.float32)   # absolute (1,)

        euler_angles = axangle2euler(rot)
        roll, pitch, yaw = euler_angles

        action = np.concatenate([world, euler_angles, grip_abs], axis=0)


        # Convert q01, q99, and mask to numpy arrays to avoid TypeError
        q01 = np.array(self.action_norm_stats["action"]["q01"], dtype=np.float32)
        q99 = np.array(self.action_norm_stats["action"]["q99"], dtype=np.float32)
        mask = np.array(self.action_norm_stats["action"]["mask"], dtype=bool)
        a_norm = np.where(
            mask,
            2.0 * (action - q01) / np.maximum(q99 - q01, 1e-8) - 1.0,
            action,
        )

        # Build prompt → tokenize → labels mask
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": self.action_tokenizer(a_norm)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Mask all but the action tokens (+ optional stop token)
        from prismatic.vla.datasets.datasets import IGNORE_INDEX  # reuse constant

        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


    def _compute_open_gripper_sequence(self, steps, initial_prev_open=0.0, sticky_gripper_num_repeat=3, validate=False):
        """Reconstruct absolute open_gripper per step from env_action['gripper'] (relative, post-sticky)."""
        out = []
        prev_open = float(initial_prev_open)
        sticky_on = False
        sticky_action = 0.0
        repeat = 0

        for t, st in enumerate(steps):
            g = float(np.asarray(st["env_action"]["gripper"]).reshape(-1)[0])  # emitted relative (after sticky)

            # Inverse of forward: open_t = prev_open - g
            open_t = float(np.clip(prev_open - g, 0.0, 1.0))
            rel = prev_open - open_t  # equals g numerically

            # Sticky trigger
            if (abs(rel) > 0.5) and (not sticky_on):
                sticky_on = True
                sticky_action = rel
                repeat = 0

            # Sticky maintenance + reset
            if sticky_on:
                repeat += 1
                if validate and abs(g - sticky_action) > 1e-6:
                    raise ValueError(
                        f"Inconsistent sticky at t={t}: got {g:.6f}, expected {sticky_action:.6f}"
                    )
                if repeat == sticky_gripper_num_repeat:
                    sticky_on = False
                    repeat = 0
                    sticky_action = 0.0

            out.append(open_t)
            prev_open = open_t

        return np.asarray(out, dtype=np.float32)


