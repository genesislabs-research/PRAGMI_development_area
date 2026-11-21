"""
hello_world_trainer.py
S-ROS First Training Run: Rote Learning with Generalization-Gated Crystallization

This is the file that makes Theo learn. It feeds Option B rote data (single
BASIC statements) through a simple encoder into TheoRecurrentExecutor, trains
on seen instances, and periodically tests on held-out instances via the
crystallization manager. When a rule class generalizes, it crystallizes.

The training loop is deliberately minimal. No Cognitive Kernel, no subsystem
chain, no neuromodulators. Just: encode input, run through LIF executor,
compute loss against expected output, backprop, check generalization, freeze
when ready. Everything else comes after this works.

WHAT WE ARE TESTING
Whether a recurrent LIF network can learn compositional rules (not just
memorize specific input-output pairs) and whether the crystallization
manager correctly detects generalization and freezes the pathway.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

from crystallization_manager import CrystallizationManager
from Theo_Core_t import spike_fn
from rote_data_generator import build_rule_classes


# =========================================================================
# TEXT ENCODER / DECODER
# =========================================================================

class CharVocab:
    """Character-level vocabulary. No external tokenizer dependency."""

    def __init__(self, max_chars: int = 128):
        self.max_chars = max_chars
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.pad_idx = 0
        self.char_to_idx["<PAD>"] = 0
        self.idx_to_char[0] = "<PAD>"
        for i in range(32, 127):
            c = chr(i)
            idx = len(self.char_to_idx)
            self.char_to_idx[c] = idx
            self.idx_to_char[idx] = c
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Encode string to padded integer tensor."""
        ids = []
        for c in text[:max_len]:
            ids.append(self.char_to_idx.get(c, self.pad_idx))
        while len(ids) < max_len:
            ids.append(self.pad_idx)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode integer tensor back to string."""
        chars = []
        for idx in ids.tolist():
            if idx == self.pad_idx:
                break
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)


# =========================================================================
# ROTE LEARNING MODEL
# =========================================================================

class RoteLearner(nn.Module):
    """
    Minimal LIF-based sequence-to-sequence model for rote BASIC learning.

    Input: character-encoded BASIC line (padded to max_len)
    Output: character-encoded semantic description (padded to max_len)

    KNOWN LIMIT: this architecture cannot generalize the rote rules, by
    construction. The whole line is flattened into one Linear layer (one weight
    per character slot, no weight sharing) and every output position is an
    independent softmax, so it has no way to copy an unseen literal such as
    'Cat' into the output. Neo will therefore, correctly, refuse to crystallize
    these rules. It is kept as the minimal harness; a learner that can pass
    needs a shared-weight sequence model with a copy path.

    Architecture: embedding -> LIF recurrent layers -> output projection
    The LIF dynamics give us the spiking substrate. The recurrence gives
    us the ability to process sequential input. The output projection
    maps back to character space for comparison with ground truth.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_lif_layers: int = 2,
        max_len: int = 128,
        beta: float = 0.7,
        num_steps: int = 15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.beta = beta
        self.num_steps = num_steps
        self.num_lif_layers = num_lif_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim * max_len, hidden_dim)

        self.lif_weights = nn.ModuleList()
        for i in range(num_lif_layers):
            self.lif_weights.append(nn.Linear(hidden_dim, hidden_dim))

        self.recurrent_weights = nn.ModuleList()
        for i in range(num_lif_layers):
            self.recurrent_weights.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        self.output_proj = nn.Linear(hidden_dim, vocab_size * max_len)

    def lif_step(
        self, v: torch.Tensor, current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single LIF neuron step: membrane update, threshold, reset."""
        v = self.beta * v + (1 - self.beta) * current
        # Surrogate gradient. A bare (v >= 1.0).float() has zero gradient, so
        # nothing below the output projection ever learned.
        spikes = spike_fn(v - 1.0)
        v = v * (1 - spikes.detach())
        return v, spikes

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, max_len) integer tensor
        returns: (batch, max_len, vocab_size) logits
        """
        batch = input_ids.shape[0]
        emb = self.embedding(input_ids)
        x = emb.view(batch, -1)
        x = self.input_proj(x)

        v = [torch.zeros(batch, self.hidden_dim, device=x.device)
             for _ in range(self.num_lif_layers)]
        spikes = [torch.zeros_like(v[0]) for _ in range(self.num_lif_layers)]

        for _ in range(self.num_steps):
            for i in range(self.num_lif_layers):
                if i == 0:
                    current = self.lif_weights[i](x) + self.recurrent_weights[i](spikes[i])
                else:
                    current = self.lif_weights[i](spikes[i - 1]) + self.recurrent_weights[i](spikes[i])
                v[i], spikes[i] = self.lif_step(v[i], current)

        out = self.output_proj(spikes[-1])
        out = out.view(batch, self.max_len, self.vocab_size)
        return out


# =========================================================================
# TRAINING LOOP
# =========================================================================

class HelloWorldTrainer:
    """
    The training rig. Feeds rote data through RoteLearner, monitors per-rule
    generalization via CrystallizationManager, freezes rules as they converge.
    """

    def __init__(
        self,
        device: str = "auto",
        lr: float = 1e-3,
        max_len: int = 128,
        eval_interval: int = 25,
        log_interval: int = 50,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_len = max_len
        self.eval_interval = eval_interval
        self.log_interval = log_interval

        self.vocab = CharVocab()
        self.model = RoteLearner(
            vocab_size=self.vocab.vocab_size,
            max_len=max_len,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # PAD is a real target. With ignore_index=0 the model was never taught
        # where the output ends, so its decoded string ran on into garbage and
        # exact match against the expected output could never succeed.
        self.criterion = nn.CrossEntropyLoss()

        self.cm = CrystallizationManager(
            loss_threshold=0.5,
            generalization_target=0.70,
            variance_threshold=1e-4,
            consecutive_windows_required=5,
            window_size=20,
            on_regression="decrystallize",
            retention_target=0.70,
        )

        self.rules = build_rule_classes()
        for rule in self.rules:
            self.cm.register_rule_class(
                rule_id=rule["rule_id"],
                description=rule["description"],
                train_pairs=rule["train_pairs"],
                test_pairs=rule["test_pairs"],
            )

        self.step_count = 0
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Rule classes: {len(self.rules)}")
        print(f"Vocab size: {self.vocab.vocab_size}")

    def encode_pair(self, pair: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an input/output pair to tensors."""
        input_ids = self.vocab.encode(pair["input"], self.max_len).unsqueeze(0).to(self.device)
        target_ids = self.vocab.encode(pair["output"], self.max_len).unsqueeze(0).to(self.device)
        return input_ids, target_ids

    def train_step(self, pair: Dict) -> float:
        """One training step on one input/output pair."""
        self.model.train()
        input_ids, target_ids = self.encode_pair(pair)
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, self.vocab.vocab_size), target_ids.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_weight_delta(self) -> float:
        """Compute norm of parameter gradients as proxy for weight delta."""
        total = 0.0
        count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
                count += 1
        return (total / max(count, 1)) ** 0.5

    def generalization_encode_fn(self, pair: Dict) -> torch.Tensor:
        """Encode function for crystallization manager generalization test."""
        return self.vocab.encode(pair["input"], self.max_len).unsqueeze(0).to(self.device)

    def generalization_decode_fn(self, logits: torch.Tensor) -> str:
        """Decode model output to string for comparison."""
        pred_ids = logits.argmax(dim=-1).squeeze(0)
        return self.vocab.decode(pred_ids)

    def generalization_compare_fn(self, predicted: str, pair: Dict) -> bool:
        """Compare predicted output to expected. Exact match for now."""
        expected = pair["output"]
        return predicted.strip() == expected.strip()

    def train_epoch(self) -> Dict:
        """One pass through all training pairs across all non-crystallized rules."""
        epoch_losses = {}
        crystallization_events = []

        # Interleave rules. Training one rule in a block and probing it straight
        # afterwards measures recency, not understanding.
        schedule = []
        for rule in self.rules:
            rid = rule["rule_id"]
            if self.cm.rule_classes[rid].is_crystallized:
                schedule.append((rid, random.choice(rule["train_pairs"]), True))   # replay
            else:
                schedule.extend((rid, pair, False) for pair in rule["train_pairs"])
        random.shuffle(schedule)

        losses_by_rule: Dict[str, List[float]] = {}
        for rid, pair, is_replay in schedule:
            loss = self.train_step(pair)
            self.step_count += 1
            if is_replay or self.cm.rule_classes[rid].is_crystallized:
                continue
            delta = self.compute_weight_delta()
            losses_by_rule.setdefault(rid, []).append(loss)

            event = self.cm.step(
                rule_id=rid,
                train_loss=loss,
                delta_norm=delta,
                model=self.model,
                encode_fn=self.generalization_encode_fn,
                decode_fn=self.generalization_decode_fn,
                compare_fn=self.generalization_compare_fn,
                eval_interval=self.eval_interval,
            )
            if event is not None:
                crystallization_events.append(event)
                print(f"\n  CRYSTALLIZED: {rid} at step {event['crystallized_at_step']}")
                print(f"    generalization: {event['final_generalization_accuracy']:.2%}")
                print(f"    train loss: {event['final_train_loss']:.4f}")

        for rid, losses in losses_by_rule.items():
            epoch_losses[rid] = sum(losses) / len(losses)

        # Re-test every crystallized rule on the current weights.
        for rid, acc in self.cm.audit_retention(
            self.model, self.generalization_encode_fn,
            self.generalization_decode_fn, self.generalization_compare_fn,
        ).items():
            if acc < self.cm.retention_target:
                print(f"\n  FORGOTTEN: {rid} retention {acc:.0%} -> returned to plastic pool")

        return {
            "losses": epoch_losses,
            "crystallizations": crystallization_events,
            "fraction_crystallized": self.cm.fraction_crystallized(),
        }

    def train(self, max_epochs: int = 500, target_crystallization: float = 1.0) -> None:
        """Main training loop. Runs until all rules crystallize or max_epochs."""
        print(f"\nStarting training: max {max_epochs} epochs, target {target_crystallization:.0%} crystallized")
        print(f"{'='*70}")

        for epoch in range(1, max_epochs + 1):
            result = self.train_epoch()

            if epoch % self.log_interval == 0 or epoch <= 5:
                active = {k: v for k, v in result["losses"].items()}
                n_active = len(active)
                avg_loss = sum(active.values()) / max(n_active, 1)
                frac = result["fraction_crystallized"]
                print(
                    f"  Epoch {epoch:4d} | active rules: {n_active:2d} | "
                    f"avg loss: {avg_loss:.4f} | crystallized: {frac:.0%} | "
                    f"steps: {self.step_count}"
                )

            if result["fraction_crystallized"] >= target_crystallization:
                print(f"\n  All rules crystallized at epoch {epoch}.")
                break
        else:
            frac = self.cm.fraction_crystallized()
            print(f"\n  Reached max epochs. {frac:.0%} crystallized.")

        print(f"\nFinal status:")
        for rid, info in self.cm.status().items():
            tag = "FROZEN" if info["crystallized"] else "PLASTIC"
            # Measured now, on the final weights.
            gen = CrystallizationManager._accuracy(
                self.cm.rule_classes[rid].test_pairs, self.model, self.generalization_encode_fn,
                self.generalization_decode_fn, self.generalization_compare_fn)
            open_gates = [g for g, ok in info["gates"].items() if not ok]
            why = "" if info["crystallized"] else f"  waiting on: {', '.join(open_gates)}"
            print(f"  [{tag:7s}] {rid:30s} held-out now:{gen:.2%}{why}")

    def save(self, path: str) -> None:
        """Save model and crystallization log."""
        torch.save({
            "model_state": self.model.state_dict(),
            "step_count": self.step_count,
            "crystallization_log": self.cm.crystallization_log,
            "status": self.cm.status(),
        }, path)
        print(f"Saved to {path}")


# =========================================================================
# SELF-TESTS
# =========================================================================

def test_char_vocab():
    v = CharVocab()
    encoded = v.encode("PRINT", max_len=10)
    assert encoded.shape == (10,)
    decoded = v.decode(encoded)
    assert decoded == "PRINT"
    print("PASS: test_char_vocab")


def test_model_forward():
    v = CharVocab()
    m = RoteLearner(vocab_size=v.vocab_size, max_len=32)
    x = v.encode("line 10: PRINT 42", max_len=32).unsqueeze(0)
    out = m(x)
    assert out.shape == (1, 32, v.vocab_size)
    print("PASS: test_model_forward")


def test_train_step():
    trainer = HelloWorldTrainer(device="cpu", max_len=64)
    pair = {"input": 'line 10: PRINT "Hello"', "output": "Print 'Hello' to screen, next line 20"}
    loss = trainer.train_step(pair)
    assert loss > 0
    print(f"PASS: test_train_step (loss={loss:.4f})")


def test_crystallization_manager_wired():
    trainer = HelloWorldTrainer(device="cpu", max_len=64)
    assert len(trainer.cm.rule_classes) == 15
    assert trainer.cm.fraction_crystallized() == 0.0
    print("PASS: test_crystallization_manager_wired")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-ROS Hello World Trainer")
    parser.add_argument("--test-only", action="store_true", help="Run tests and exit")
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", type=str, default=None, help="Path to save checkpoint")
    args = parser.parse_args()

    if args.test_only:
        test_char_vocab()
        test_model_forward()
        test_train_step()
        test_crystallization_manager_wired()
        print("\nAll hello_world_trainer tests passed.")
    else:
        trainer = HelloWorldTrainer(device=args.device, lr=args.lr, max_len=128)
        trainer.train(max_epochs=args.epochs)
        if args.save:
            trainer.save(args.save)
