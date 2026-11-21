"""
stage_zero_keys.py
Stage Zero (Motor Babbling): Neo gates each key into Theo's S-CAM

This is the closed loop the README describes and that no other file in S-ROS
implements: the plastic executor learns a key, Neo decides whether it has
really been learned, and only then is the (sensor, action) pair written into
Theo's S-CAM as a frozen engram. Inference order is S-CAM first, executor
second. Keys are trained sequentially, one at a time.

WHAT IS BEING CLAIMED, AND HOW EACH CLAIM IS CHECKED

  1. Neo's gate is honest. A key crystallizes only when the executor answers
     UNSEEN noisy presentations of it at 100% for K windows, its training
     loss is low, its learning signal has decayed, and it does not give this
     key's answer to other keys.
  2. Crystallizing writes real memory. On crystallization Neo calls
     commit_fn, which writes the engram into the S-CAM.
  3. The memory is durable. After every key is frozen, the executor is
     deliberately trained to give WRONG answers for all of them. The
     executor alone then fails, and the full path (S-CAM first) still answers
     every key. The S-CAM content hash is unchanged.
  4. The memory survives a restart. The core is saved to a .soul file, loaded
     into a fresh TheoCore, and audited again.

If any of these fails the script exits non-zero and says which.

ENCODING
A key is a fixed population code: CODE_BITS of the 128 sensor lines are
active. A "presentation" is that code under sensor noise (a dropped bit, a
stray bit). Training presentations and held-out presentations are disjoint.
The action for key N is a spike on output line N (README: "output spike
position N corresponds to key N"). The S-CAM recalls by overlap, so a noisy
presentation still retrieves its engram once the key is frozen.

Usage:
    python3 stage_zero_keys.py                 # all 39 keys
    python3 stage_zero_keys.py --keys 8        # quick run
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crystallization_manager import CrystallizationManager
from Theo_Core_t import TheoCore
from theo_checkpoint import save_theo_soul, load_theo_soul

SPIKE_DIM = 128
COORD_DIM = 64
CODE_BITS = 8
KEYS: List[str] = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + ["SPACE", "COMMA", "ENTER"]
)
assert len(KEYS) == 39


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_key_codes(n_keys: int, gen: torch.Generator) -> torch.Tensor:
    """One fixed CODE_BITS-of-128 population code per key. Pairwise overlap is kept low."""
    codes: List[torch.Tensor] = []
    while len(codes) < n_keys:
        c = torch.zeros(SPIKE_DIM)
        c[torch.randperm(SPIKE_DIM, generator=gen)[:CODE_BITS]] = 1.0
        if all(float((c * o).sum()) <= 2 for o in codes):
            codes.append(c)
    return torch.stack(codes)


def noisy(code: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Sensor noise: drop one code bit, and/or add one stray bit. Never the clean code."""
    x = code.clone()
    mode = int(torch.randint(0, 3, (1,), generator=gen))
    if mode in (0, 2):
        on = torch.nonzero(x).flatten()
        x[on[int(torch.randint(0, len(on), (1,), generator=gen))]] = 0.0
    if mode in (1, 2):
        off = torch.nonzero(code == 0).flatten()
        x[off[int(torch.randint(0, len(off), (1,), generator=gen))]] = 1.0
    return x


def build_pairs(codes: torch.Tensor, n_train: int, n_test: int,
                gen: torch.Generator) -> Dict[int, Tuple[List[Dict], List[Dict]]]:
    """Per key: training presentations (clean + noisy) and disjoint held-out noisy ones."""
    data = {}
    for k in range(codes.shape[0]):
        seen = {tuple(codes[k].tolist())}
        train = [{"input": tuple(codes[k].tolist()), "output": k}]
        test: List[Dict] = []
        while len(train) < n_train or len(test) < n_test:
            x = tuple(noisy(codes[k], gen).tolist())
            if x in seen:
                continue
            seen.add(x)
            (train if len(train) < n_train else test).append({"input": x, "output": k})
        data[k] = (train, test)
    return data


# ---------------------------------------------------------------------------
# Inference paths Neo is allowed to look through
# ---------------------------------------------------------------------------
class FullPath(nn.Module):
    """The real inference path: S-CAM first, executor second. Returns action spikes."""
    def __init__(self, theo: TheoCore):
        super().__init__()
        self.theo = theo
        self.register_buffer("coords", torch.zeros(1, COORD_DIM))

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        return self.theo(sensor, self.coords).action_spikes


class ExecutorOnly(nn.Module):
    """The plastic executor with the S-CAM bypassed. Used only to measure forgetting."""
    def __init__(self, theo: TheoCore):
        super().__init__()
        self.theo = theo

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        logits, _ = self.theo.executor_logits(sensor, torch.zeros(sensor.shape[0], COORD_DIM))
        return (logits > 0).float()


def encode_fn(pair: Dict) -> torch.Tensor:
    return torch.tensor(pair["input"]).unsqueeze(0)


def decode_fn(action_spikes: torch.Tensor) -> int:
    """Exactly one action line firing -> that key. Anything else -> -1 (no key pressed)."""
    on = torch.nonzero(action_spikes.flatten() > 0).flatten()
    return int(on[0]) if on.numel() == 1 else -1


def compare_fn(predicted: int, pair: Dict) -> bool:
    return predicted == pair["output"]


def accuracy(model: nn.Module, pairs: List[Dict]) -> float:
    with torch.no_grad():
        return sum(compare_fn(decode_fn(model(encode_fn(p))), p) for p in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Stage Zero
# ---------------------------------------------------------------------------
def run(n_keys: int, max_steps_per_key: int, seed: int, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    codes = build_key_codes(n_keys, gen)
    data = build_pairs(codes, n_train=16, n_test=8, gen=gen)

    theo = TheoCore(spike_dim=SPIKE_DIM, coord_dim=COORD_DIM, max_engrams=1024,
                    num_steps=10, match_threshold=0.7)
    full = FullPath(theo)
    plastic = ExecutorOnly(theo)
    opt = torch.optim.Adam(theo.parameters(), lr=3e-3)

    def commit(rule_id: str, rc) -> None:
        k = rc.train_pairs[0]["output"]
        action = torch.zeros(SPIKE_DIM)
        action[k] = 1.0
        theo.crystallize_from_plastic(codes[k], action)

    neo = CrystallizationManager(
        loss_threshold=0.02,
        generalization_target=1.0,          # README: held-out at 100%
        variance_threshold=1e-4,
        consecutive_windows_required=3,     # README: 3 consecutive windows
        window_size=20,
        variance_check_quantity="gradient",
        retention_target=1.0,
        on_regression="flag",
        commit_fn=commit,
    )
    for k in range(n_keys):
        neo.register_rule_class(f"key_{KEYS[k]}", f"press {KEYS[k]}", *data[k])

    t0 = time.time()
    silence = torch.zeros(SPIKE_DIM)
    for k in range(n_keys):
        rid = f"key_{KEYS[k]}"
        x_pos = torch.tensor([p["input"] for p in data[k][0]])
        y_pos = torch.zeros(len(x_pos), SPIKE_DIM)
        y_pos[:, k] = 1.0
        # Contrast set: keys not learned yet must produce NO key press, so that
        # "press K" is specific to K. Frozen keys need no rehearsal: the S-CAM
        # answers for them.
        later = [p for j in range(k + 1, n_keys) for p in data[j][0]]
        event = None
        for step in range(max_steps_per_key):
            idx = torch.randint(0, len(x_pos), (8,), generator=gen)
            xb, yb = x_pos[idx], y_pos[idx]
            if later:
                nidx = torch.randint(0, len(later), (8,), generator=gen)
                xn = torch.tensor([later[int(i)]["input"] for i in nidx])
                xb = torch.cat([xb, xn])
                yb = torch.cat([yb, silence.expand(len(xn), -1)])
            opt.zero_grad()
            logits, _ = theo.executor_logits(xb, torch.zeros(len(xb), COORD_DIM))
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            grad = neo._compute_gradient_rms_norm(theo)
            opt.step()
            event = neo.step(rid, loss.item(), grad, full, encode_fn, decode_fn, compare_fn,
                             eval_interval=10)
            if event is not None:
                break
        if event is None:
            gates = neo.gate_report(rid)
            raise SystemExit(
                f"STAGE ZERO FAILED: key {KEYS[k]} did not crystallize in {max_steps_per_key} steps. "
                f"Gates: {gates}. README: no later stage may run on a system missing a base key."
            )
        retention = neo.audit_retention(full, encode_fn, decode_fn, compare_fn)
        if verbose:
            print(f"  [{k + 1:2d}/{n_keys}] {KEYS[k]:>5s} crystallized at step {event['crystallized_at_step']:5d} "
                  f"| held-out {event['final_generalization_accuracy']:.0%} "
                  f"| engrams {theo.scam.inspect_engrams()['active_count']:2d} "
                  f"| retention of all frozen keys {min(retention.values()):.0%}")

    all_pairs = [p for k in range(n_keys) for p in data[k][0] + data[k][1]]
    results: Dict = {"n_keys": n_keys, "train_seconds": time.time() - t0,
                     "total_steps": neo.global_step}

    # --- Claim 1 & 2: every key crystallized and is in the S-CAM
    results["fraction_crystallized"] = neo.fraction_crystallized()
    results["engrams"] = theo.scam.inspect_engrams()["active_count"]

    # --- Before interference
    results["full_path_before"] = accuracy(full, all_pairs)
    results["executor_only_before"] = accuracy(plastic, all_pairs)
    hash_before = theo.scam.content_hash()

    # --- Claim 3: deliberately un-teach the executor
    x_all = torch.tensor([p["input"] for p in all_pairs])
    wrong = torch.zeros(len(all_pairs), SPIKE_DIM)
    for i, p in enumerate(all_pairs):
        wrong[i, (p["output"] + 1) % n_keys] = 1.0      # the next key's action: always wrong
    for _ in range(300):
        idx = torch.randint(0, len(x_all), (32,), generator=gen)
        opt.zero_grad()
        logits, _ = theo.executor_logits(x_all[idx], torch.zeros(32, COORD_DIM))
        F.binary_cross_entropy_with_logits(logits, wrong[idx]).backward()
        opt.step()
    results["executor_only_after"] = accuracy(plastic, all_pairs)
    results["full_path_after"] = accuracy(full, all_pairs)
    results["scam_hash_unchanged"] = theo.scam.content_hash() == hash_before
    retention = neo.audit_retention(full, encode_fn, decode_fn, compare_fn)
    results["retention_after_interference"] = min(retention.values())
    results["regressions"] = len(neo.regression_log)

    # --- Claim 4: restart
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stage_zero.soul")
        save_theo_soul(theo, neo.crystallization_log, path)
        fresh = TheoCore(spike_dim=SPIKE_DIM, coord_dim=COORD_DIM, max_engrams=1024,
                         num_steps=10, match_threshold=0.7)
        ledger, _ = load_theo_soul(path, fresh)
        results["full_path_after_reload"] = accuracy(FullPath(fresh), all_pairs)
        results["ledger_entries_after_reload"] = len(ledger)
    return results


def verdict(r: Dict) -> List[str]:
    """Return the list of failed claims (empty = Stage Zero proven)."""
    failures = []
    if r["fraction_crystallized"] != 1.0 or r["engrams"] != r["n_keys"]:
        failures.append("not every key crystallized into the S-CAM")
    if r["full_path_before"] != 1.0:
        failures.append("full path was not 100% before interference")
    if r["executor_only_after"] > 0.2:
        failures.append("interference did not actually un-teach the executor, so durability was not tested")
    if r["full_path_after"] != 1.0 or r["retention_after_interference"] != 1.0 or r["regressions"]:
        failures.append("frozen keys were lost when the executor was retrained")
    if not r["scam_hash_unchanged"]:
        failures.append("S-CAM content changed during training")
    if r["full_path_after_reload"] != 1.0 or r["ledger_entries_after_reload"] != r["n_keys"]:
        failures.append("memory did not survive save/load")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S-ROS Stage Zero: Neo-gated key engrams in Theo's S-CAM")
    ap.add_argument("--keys", type=int, default=39, help="number of keys (max 39)")
    ap.add_argument("--max-steps-per-key", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)

    n = min(args.keys, len(KEYS))
    print(f"Stage Zero: {n} keys, trained sequentially, Neo gating each into the S-CAM\n")
    r = run(n, args.max_steps_per_key, args.seed)

    print(f"\n{r['total_steps']} training steps in {r['train_seconds']:.0f}s")
    print(f"Keys crystallized into S-CAM ............ {r['engrams']}/{r['n_keys']}")
    print(f"Before interference  full path .......... {r['full_path_before']:.0%}")
    print(f"                     executor alone ..... {r['executor_only_before']:.0%}")
    print(f"After un-teaching    executor alone ..... {r['executor_only_after']:.0%}")
    print(f"                     full path .......... {r['full_path_after']:.0%}")
    print(f"S-CAM content hash unchanged ............ {r['scam_hash_unchanged']}")
    print(f"Neo retention audit (worst key) ......... {r['retention_after_interference']:.0%}, "
          f"{r['regressions']} regressions")
    print(f"After save -> fresh core -> load ........ {r['full_path_after_reload']:.0%}")

    failed = verdict(r)
    if failed:
        print("\nSTAGE ZERO FAILED:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSTAGE ZERO PASSED: every key was earned, frozen, and kept.")
