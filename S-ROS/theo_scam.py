"""
theo_scam.py  (was Theo_S-CAM_t.py; renamed because a hyphen cannot be imported)
Spiking Content-Addressable Memory (64K Procedural RAM)

COMPUTATIONAL GROUNDING:
The frozen engram store of the deterministic core. It hybridizes associative
memory (content-addressable lookup) with sequential execution (program counter),
allowing S-ROS to store procedural logic natively.

This is the ONE canonical S-CAM. Theo_Core_t.py imports it; there is no second
copy. The contract it guarantees, and which its self-test asserts:

  1. EXACT MATCH. A query hits an engram only if its match score reaches
     match_threshold. The score is overlap / max(|query|, |key|), which is 1.0
     only when the two binary patterns are identical. Subsets and supersets of
     a key do not score 1.0.
  2. HONEST MISS. A miss returns an all-zero action, score < threshold, and
     index -1. It never returns some other engram's action.
  3. WRITE ONCE. Writing a key that is already stored with the same action is a
     no-op. Writing it with a different action raises unless overwrite=True.
     There are never two slots for one key.
  4. NO EVICTION. A full memory raises. A crystallized engram is never
     silently replaced.
  5. NOT GRADIENT REACHABLE. Everything is a registered buffer, so no optimizer
     can drift an engram.

Primary grounding papers:
Hopfield JJ (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS. DOI: 10.1073/pnas.79.8.2554
Sun Z et al. (2025). "General and stable emulation of finite state machines with spiking neural networks." Neuromorphic Computing and Engineering. DOI: 10.1088/2634-4386/adc32d
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class SCAMFullError(RuntimeError):
    """Raised when a write would require evicting a crystallized engram."""


class SCAMConflictError(ValueError):
    """Raised when a key is rewritten with a different action without overwrite=True."""


class TheoSCAM(nn.Module):
    """
    Deterministic program memory: binary sensor key -> binary action value.

    Capacity note: with strictly one-hot keys there are only spike_dim distinct
    keys, so slots beyond spike_dim are only reachable with multi-bit keys or by
    sequential (program counter) addressing.
    """

    def __init__(self, spike_dim: int = 128, max_engrams: int = 65536,
                 match_threshold: float = 0.999):
        super().__init__()
        self.spike_dim = spike_dim
        self.max_engrams = max_engrams
        self.match_threshold = match_threshold

        self.register_buffer("sensor_keys", torch.zeros(max_engrams, spike_dim, dtype=torch.bool))
        self.register_buffer("action_values", torch.zeros(max_engrams, spike_dim, dtype=torch.bool))
        self.register_buffer("is_active", torch.zeros(max_engrams, dtype=torch.bool))
        self.register_buffer("usage_counts", torch.zeros(max_engrams, dtype=torch.long))
        self.register_buffer("program_counter", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ helpers
    def _as_pattern(self, spikes: torch.Tensor, label: str) -> torch.Tensor:
        """Validate and binarize to a [B, spike_dim] bool tensor on our device."""
        if spikes.shape[-1] != self.spike_dim:
            raise ValueError(
                f"[Theo S-CAM] {label} shape mismatch. Expected {self.spike_dim}, got {spikes.shape[-1]}"
            )
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)
        if spikes.dim() != 2:
            raise ValueError(f"[Theo S-CAM] {label} must be [spike_dim] or [B, spike_dim], got {tuple(spikes.shape)}")
        return (spikes.to(self.sensor_keys.device) > 0)

    def _single(self, spikes: torch.Tensor, label: str) -> torch.Tensor:
        pat = self._as_pattern(spikes, label)
        if pat.shape[0] != 1:
            raise ValueError(f"[Theo S-CAM] {label} must be a single pattern, got batch of {pat.shape[0]}")
        return pat[0]

    # ------------------------------------------------------------------- lookup
    @torch.no_grad()
    def lookup(self, sensor_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pure batched associative lookup. No side effects.

        Returns:
            actions [B, spike_dim] float  (all zeros on a miss)
            scores  [B] float             (best match score, 0.0 if memory empty)
            indices [B] long              (slot address, -1 on a miss)
        """
        query = self._as_pattern(sensor_spikes, "sensor")
        B = query.shape[0]
        device = self.sensor_keys.device
        actions = torch.zeros(B, self.spike_dim, device=device)
        scores = torch.zeros(B, device=device)
        indices = torch.full((B,), -1, dtype=torch.long, device=device)

        slots = torch.nonzero(self.is_active, as_tuple=False).flatten()
        if slots.numel() == 0:
            return actions, scores, indices

        keys = self.sensor_keys[slots].float()                    # [N, D]
        q = query.float()                                         # [B, D]
        overlap = q @ keys.T                                      # [B, N]
        q_norm = q.sum(dim=-1, keepdim=True)                      # [B, 1]
        k_norm = keys.sum(dim=-1).unsqueeze(0)                    # [1, N]
        denom = torch.maximum(q_norm, k_norm).clamp_min(1.0)
        score = overlap / denom                                   # 1.0 iff identical

        best_score, best_pos = score.max(dim=-1)
        hit = best_score >= self.match_threshold
        scores.copy_(best_score)
        hit_slots = slots[best_pos[hit]]
        indices[hit] = hit_slots
        actions[hit] = self.action_values[hit_slots].float()
        return actions, scores, indices

    @torch.no_grad()
    def retrieve_engram(self, sensor_spikes: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        Single-pattern associative lookup for context shifts, interrupts, or jumps.
        On a hit, bumps the usage count and snaps the program counter to the slot.
        Returns (action [same shape as input], score, index). Index is -1 on a miss.
        """
        self._single(sensor_spikes, "sensor")
        actions, scores, indices = self.lookup(sensor_spikes)
        idx = int(indices[0].item())
        if idx >= 0:
            self.usage_counts[idx] += 1
            self.program_counter.fill_(idx)
        action = actions[0] if sensor_spikes.dim() == 1 else actions
        return action, float(scores[0].item()), idx

    # Name used by the old minimal S-CAM in Theo_Core_t.py.
    external_retrieve = retrieve_engram

    # --------------------------------------------------------------- sequential
    @torch.no_grad()
    def step_execution(self) -> Tuple[torch.Tensor, bool]:
        """
        Sequential execution: advance to the next address.
        Returns (action, valid). valid is False at the end of the program, so an
        end-of-program is distinguishable from a stored all-zero action.
        """
        next_address = int((self.program_counter.item() + 1) % self.max_engrams)
        if bool(self.is_active[next_address]):
            self.program_counter.fill_(next_address)
            return self.action_values[next_address].float(), True
        return torch.zeros(self.spike_dim, device=self.action_values.device), False

    # -------------------------------------------------------------------- write
    @torch.no_grad()
    def crystallize_skill(self, sensor_pattern: torch.Tensor, action_pattern: torch.Tensor,
                          address: Optional[int] = None, overwrite: bool = False) -> int:
        """
        Write one engram. Returns the slot address it lives at.

        A key is stored at most once. Rewriting an identical (key, action) pair is
        a no-op. Changing the action of an existing key, or POKEing over an
        occupied address, requires overwrite=True.
        """
        key = self._single(sensor_pattern, "sensor")
        val = self._single(action_pattern, "action")
        if not bool(key.any()):
            raise ValueError("[Theo S-CAM] refusing to store an all-zero sensor key")

        existing = self._find_exact(key)
        if existing >= 0 and (address is None or address == existing):
            if torch.equal(self.action_values[existing], val):
                return existing
            if not overwrite:
                raise SCAMConflictError(
                    f"[Theo S-CAM] key already crystallized at slot {existing} with a different action"
                )
            self.action_values[existing] = val
            return existing
        if existing >= 0 and address is not None and address != existing:
            raise SCAMConflictError(
                f"[Theo S-CAM] key already crystallized at slot {existing}; cannot duplicate at {address}"
            )

        if address is None:
            free = torch.nonzero(~self.is_active, as_tuple=False).flatten()
            if free.numel() == 0:
                raise SCAMFullError(
                    f"[Theo S-CAM] all {self.max_engrams} slots hold crystallized engrams; refusing to evict"
                )
            idx = int(free[0].item())
        else:
            if address >= self.max_engrams or address < 0:
                raise IndexError(f"Address {address} out of bounds for {self.max_engrams}-slot S-CAM.")
            if bool(self.is_active[address]) and not overwrite:
                raise SCAMConflictError(f"[Theo S-CAM] address {address} is occupied")
            idx = address

        self.sensor_keys[idx] = key
        self.action_values[idx] = val
        self.is_active[idx] = True
        self.usage_counts[idx] = 0
        return idx

    def _find_exact(self, key: torch.Tensor) -> int:
        slots = torch.nonzero(self.is_active, as_tuple=False).flatten()
        if slots.numel() == 0:
            return -1
        same = (self.sensor_keys[slots] == key.unsqueeze(0)).all(dim=-1)
        pos = torch.nonzero(same, as_tuple=False).flatten()
        return int(slots[pos[0]].item()) if pos.numel() else -1

    # --------------------------------------------------------------- inspection
    def content_hash(self) -> str:
        """SHA-256 over the stored engrams. Used by checkpoints to detect tampering or drift."""
        h = hashlib.sha256()
        for buf in (self.is_active, self.sensor_keys, self.action_values):
            h.update(buf.detach().cpu().to(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def inspect_engrams(self) -> Dict[str, Any]:
        active_count = int(self.is_active.sum().item())
        return {
            "status": "EMPTY" if active_count == 0 else "NOMINAL",
            "active_count": active_count,
            "max_capacity": self.max_engrams,
            "program_counter": int(self.program_counter.item()),
        }


# ============================== SELF-TEST ==============================
def _onehot(i: int, d: int = 128) -> torch.Tensor:
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def run_self_test() -> None:
    D = 128
    scam = TheoSCAM(D, max_engrams=64)

    # empty memory is a miss
    a, s, i = scam.retrieve_engram(_onehot(3))
    assert i == -1 and s == 0.0 and a.sum() == 0, "empty memory must miss"

    # 1. write/read round trip for 39 keys (Stage Zero keyboard)
    for k in range(39):
        scam.crystallize_skill(_onehot(k), _onehot((k * 7 + 5) % D))
    for k in range(39):
        a, s, i = scam.retrieve_engram(_onehot(k))
        assert i == k and s == 1.0, f"key {k} did not hit its own slot"
        assert torch.equal(a, _onehot((k * 7 + 5) % D)), f"key {k} returned the wrong action"
    print("PASS 1: 39/39 engrams read back exactly")

    # 2. honest miss
    a, s, i = scam.retrieve_engram(_onehot(100))
    assert i == -1 and a.sum() == 0, "unseen key must miss with a zero action"
    print("PASS 2: unseen key misses (idx -1, zero action)")

    # 3. subsets / supersets are not exact matches
    multi = torch.zeros(D); multi[[60, 61, 62]] = 1.0
    scam.crystallize_skill(multi, _onehot(9))
    sub = _onehot(60)
    sup = multi.clone(); sup[[63, 64, 65]] = 1.0
    assert scam.retrieve_engram(sub)[2] == -1, "1-of-3-bit subset must not hit"
    assert scam.retrieve_engram(sup)[2] == -1, "superset must not hit"
    assert scam.retrieve_engram(multi)[2] == 39, "exact multi-bit key must hit"
    print("PASS 3: subset and superset queries miss; exact multi-bit key hits")

    # 4. write-once semantics
    n_before = scam.inspect_engrams()["active_count"]
    assert scam.crystallize_skill(_onehot(0), _onehot(5)) == 0, "identical rewrite must be a no-op"
    assert scam.inspect_engrams()["active_count"] == n_before, "identical rewrite created a duplicate"
    try:
        scam.crystallize_skill(_onehot(0), _onehot(77))
        raise AssertionError("conflicting rewrite must raise")
    except SCAMConflictError:
        pass
    assert torch.equal(scam.retrieve_engram(_onehot(0))[0], _onehot(5)), "conflict changed the engram"
    scam.crystallize_skill(_onehot(0), _onehot(77), overwrite=True)
    assert torch.equal(scam.retrieve_engram(_onehot(0))[0], _onehot(77)), "explicit overwrite ignored"
    assert scam.inspect_engrams()["active_count"] == n_before
    scam.crystallize_skill(_onehot(0), _onehot(5), overwrite=True)
    print("PASS 4: one slot per key; conflicts raise; explicit overwrite works")

    # 5. no eviction when full
    small = TheoSCAM(D, max_engrams=4)
    for k in range(4):
        small.crystallize_skill(_onehot(k), _onehot(k + 10))
    try:
        small.crystallize_skill(_onehot(50), _onehot(51))
        raise AssertionError("full memory must raise")
    except SCAMFullError:
        pass
    for k in range(4):
        assert torch.equal(small.retrieve_engram(_onehot(k))[0], _onehot(k + 10)), "engram lost when full"
    print("PASS 5: full memory raises and keeps every engram")

    # 6. batched lookup, mixed hits and misses
    batch = torch.stack([_onehot(1), _onehot(100), _onehot(2)])
    acts, scores, idxs = scam.lookup(batch)
    assert idxs.tolist() == [1, -1, 2] and acts[1].sum() == 0
    print("PASS 6: batched lookup returns per-row hit/miss")

    # 7. program counter
    scam.retrieve_engram(_onehot(0))
    act, ok = scam.step_execution()
    assert ok and scam.program_counter.item() == 1 and torch.equal(act, _onehot(12))
    end = TheoSCAM(D, max_engrams=8)
    end.crystallize_skill(_onehot(0), _onehot(1))
    end.retrieve_engram(_onehot(0))
    assert end.step_execution()[1] is False, "stepping past the program must report invalid"
    print("PASS 7: program counter snaps on hit, steps, and reports end of program")

    # 8. engrams are not gradient reachable and survive unrelated training
    assert len(list(scam.parameters())) == 0, "S-CAM must expose no trainable parameters"
    before = scam.content_hash()
    net = nn.Linear(D, D)
    opt = torch.optim.SGD(list(net.parameters()) + list(scam.parameters()), lr=1.0)
    for _ in range(20):
        opt.zero_grad()
        net(torch.randn(8, D)).pow(2).mean().backward()
        opt.step()
    assert scam.content_hash() == before, "engrams changed during training"
    print("PASS 8: no trainable parameters; content hash unchanged by training")

    # 9. state_dict round trip
    clone = TheoSCAM(D, max_engrams=64)
    clone.load_state_dict(scam.state_dict())
    assert clone.content_hash() == scam.content_hash()
    for k in range(39):
        assert torch.equal(clone.retrieve_engram(_onehot(k))[0], scam.retrieve_engram(_onehot(k))[0])
    print("PASS 9: state_dict round trip preserves every retrieval")

    print("\nAll S-CAM self-tests passed.")


if __name__ == "__main__":
    run_self_test()
