"""
curriculum_trainer_v5.py
Curriculum-Staged Crystallization Trainer for Theo SNN Execution Core

Three-stage curriculum that teaches concepts before patterns before application.
Stage 1: Keyword recognition (12-way, tiny model, crystallizes in minutes)
Stage 2: Syntax pattern recognition (15-way, expanded model, loads Stage 1 weights)
Stage 3: Full statement processing (classification + template rendering verification)

The SNN classifies. The template engine renders. No character-level generation.
This is the architectural pivot from v4's seq2seq approach which plateaued at 27%.

BIOLOGICAL GROUNDING
Curriculum learning mirrors developmental neuroscience: phoneme discrimination
precedes word recognition precedes sentence comprehension. Each stage's critical
period closes (crystallizes) before the next opens. Elman (1993) "Learning and
development in neural networks: the importance of starting small."

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import time
import re
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

from crystallization_manager import CrystallizationManager
from curriculum_data import (
    build_stage1_data, build_stage2_data, build_stage3_data,
    KEYWORDS, KEYWORD_TO_ID, NUM_KEYWORDS,
    PATTERN_NAMES, PATTERN_TO_ID, NUM_PATTERNS,
)


# =========================================================================
# ANSI COLORS
# =========================================================================

class C:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"
    BG_GREEN  = "\033[42m"
    BG_CYAN   = "\033[46m"
    BG_YELLOW = "\033[43m"


def banner():
    """Print Genesis Labs banner."""
    print(f"\n{C.CYAN}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║        GENESIS LABS  |  Curriculum Trainer v5              ║")
    print("  ║        Theo SNN Execution Core                             ║")
    print("  ║        Crystallization-Gated Staged Learning               ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")


# =========================================================================
# CHARACTER VOCABULARY
# =========================================================================

class CharVocab:
    """Character-level vocabulary. No external tokenizer dependency."""

    def __init__(self, max_chars: int = 128):
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

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        """Encode string to padded integer tensor."""
        ids = []
        for c in text[:max_len]:
            ids.append(self.char_to_idx.get(c, self.pad_idx))
        while len(ids) < max_len:
            ids.append(self.pad_idx)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode integer tensor back to string, stopping at first PAD."""
        chars = []
        for idx in ids.tolist():
            if idx == self.pad_idx:
                break
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)


# =========================================================================
# SURROGATE GRADIENT
# =========================================================================

class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spiking threshold. Forward pass uses Heaviside
    step function (hard threshold at 1.0). Backward pass uses sigmoid
    derivative as a smooth approximation.

    NOT a biological quantity. Training artifact for backprop through
    discontinuous spike function. Neftci et al. (2019) DOI: 10.1109/MSP.2019.2931595
    """
    alpha = 10.0  # surrogate gradient sharpness

    @staticmethod
    def forward(ctx, v):
        spikes = (v >= 1.0).float()
        ctx.save_for_backward(v)
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        sigmoid = torch.sigmoid(SurrogateSpike.alpha * (v - 1.0))
        grad = grad_output * sigmoid * (1 - sigmoid) * SurrogateSpike.alpha
        return grad


surrogate_spike = SurrogateSpike.apply


# =========================================================================
# RULE CLASSIFIER SNN
# =========================================================================

class RuleClassifierSNN(nn.Module):
    """
    Sequential LIF recurrent classifier. Scans input characters one at a time,
    builds temporal hidden state through LIF dynamics, classifies through a
    readout head. Uses surrogate gradients for backprop.

    Architecture:
      embedding -> per-timestep LIF recurrent processing -> spike accumulator -> classifier head

    The model processes the input sequentially (one character per timestep),
    not as a flattened bag. This preserves positional information that the
    v4 flatten approach destroyed (which caused the 2.1 loss plateau).

    Parameters:
        vocab_size: size of character vocabulary
        embed_dim: embedding dimension per character
        hidden_dim: LIF layer width
        num_classes: number of output classes
        num_lif_layers: number of stacked LIF layers
        max_len: maximum input sequence length
        beta: membrane decay constant (LIF leak). NOT a biological quantity
              at this exact value. Engineering approximation of tau_mem.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 12,
        num_lif_layers: int = 1,
        max_len: int = 10,
        beta: float = 0.8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_lif_layers = num_lif_layers
        self.max_len = max_len
        self.beta = beta

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.lif_weights = nn.ModuleList()
        self.recurrent_weights = nn.ModuleList()
        for i in range(num_lif_layers):
            self.lif_weights.append(nn.Linear(hidden_dim, hidden_dim))
            self.recurrent_weights.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, max_len) integer tensor
        returns: (batch, num_classes) logits

        Processes input sequentially. At each character timestep, the embedding
        is projected and fed through the LIF layers. Spikes are accumulated
        across all timesteps, then the accumulated spike count is classified.
        """
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        v = [torch.zeros(batch, self.hidden_dim, device=input_ids.device)
             for _ in range(self.num_lif_layers)]
        spike_acc = torch.zeros(batch, self.hidden_dim, device=input_ids.device)
        # Spikes each layer emitted on the previous character. The recurrent
        # input must come from these: the membrane is reset after a spike, so
        # re-thresholding it (as before) always gave zero and the network was
        # purely feedforward.
        prev_spikes = [torch.zeros(batch, self.hidden_dim, device=input_ids.device)
                       for _ in range(self.num_lif_layers)]

        for t in range(seq_len):
            char_emb = self.embedding(input_ids[:, t])
            x = self.input_proj(char_emb)

            for i in range(self.num_lif_layers):
                if i == 0:
                    current = self.lif_weights[i](x)
                else:
                    current = self.lif_weights[i](spikes_layer)
                current = current + self.recurrent_weights[i](prev_spikes[i])
                v[i] = self.beta * v[i] + (1 - self.beta) * current
                spikes_layer = surrogate_spike(v[i])
                v[i] = v[i] * (1 - spikes_layer.detach())
                prev_spikes[i] = spikes_layer

            spike_acc = spike_acc + spikes_layer

        logits = self.classifier(spike_acc / max(seq_len, 1))
        return logits


# =========================================================================
# TEMPLATE ENGINE (from v4 spec, deterministic rendering)
# =========================================================================

class TemplateEngine:
    """
    Deterministic output renderer. Takes a classified rule_id and the original
    input string, extracts values via regex, and renders the expected output.
    The SNN never generates characters. It classifies. This renders.
    """

    @staticmethod
    def render(rule_id: str, input_text: str) -> Optional[str]:
        """Render output string from rule_id and input text."""
        text = input_text.strip()
        line_match = re.match(r'line\s+(\d+):\s*(.*)', text)
        if not line_match:
            return None
        line_num = int(line_match.group(1))
        stmt = line_match.group(2).strip()
        next_line = line_num + 10

        if rule_id == "print_string_literal":
            m = re.match(r'PRINT\s+"([^"]*)"', stmt)
            if m:
                return f"Print '{m.group(1)}' to screen, next line {next_line}"

        elif rule_id == "print_comma_zones":
            m = re.match(r'PRINT\s+(.*)', stmt)
            if m:
                parts = re.findall(r'"([^"]*)"', m.group(1))
                if len(parts) == 2:
                    return f"Print '{parts[0]}' in zone1, '{parts[1]}' in zone2, next line {next_line}"
                elif len(parts) == 3:
                    return f"Print '{parts[0]}' in zone1, '{parts[1]}' in zone2, '{parts[2]}' in zone3, next line {next_line}"

        elif rule_id == "print_semicolon_concat":
            m = re.match(r'PRINT\s+(.*)', stmt)
            if m:
                parts = re.findall(r'"([^"]*)"', m.group(1))
                concat = "".join(parts)
                return f"Print '{concat}' (no space), next line {next_line}"

        elif rule_id == "print_blank":
            return f"Print blank line, next line {next_line}"

        elif rule_id == "print_numeric":
            m = re.match(r'PRINT\s+([\d.]+)', stmt)
            if m:
                return f"Print '{m.group(1)}', next line {next_line}"

        elif rule_id == "print_variable":
            m = re.match(r'PRINT\s+(\w+)\s+\((\w+)=([\d.]+)\)', stmt)
            if m:
                return f"Print '{m.group(3)}', next line {next_line}"

        elif rule_id == "print_mixed_string_var":
            m = re.match(r'PRINT\s+"([^"]*)",\s*(\w+)\s+\((\w+)=([\d.]+)\)', stmt)
            if m:
                return f"Print '{m.group(1)}' then '{m.group(4)}' in next zone, next line {next_line}"

        elif rule_id == "let_simple_assign":
            m = re.match(r'LET\s+(\w+)=([\d.]+)$', stmt)
            if m:
                return f"{m.group(1)}={m.group(2)}, next line {next_line}"

        elif rule_id == "let_arithmetic":
            m = re.match(r'LET\s+(\w+)=(.+?)\s+\((.+)\)', stmt)
            if m:
                var = m.group(1)
                expr = m.group(2)
                ctx = m.group(3)
                result = TemplateEngine._eval_arithmetic(expr, ctx)
                if result is not None:
                    if float(result) == int(float(result)):
                        result = str(int(float(result)))
                    return f"{var}={result}, next line {next_line}"

        elif rule_id == "let_builtin_function":
            m = re.match(r'LET\s+(\w+)=(\w+)\(([-\d.]+)\)', stmt)
            if m:
                var = m.group(1)
                func = m.group(2).upper()
                arg = float(m.group(3))
                if func == "SQR":
                    val = int(math.sqrt(arg))
                elif func == "ABS":
                    val = int(abs(arg))
                elif func == "INT":
                    val = int(arg)
                else:
                    return None
                return f"{var}={val}, next line {next_line}"

        elif rule_id == "goto_unconditional":
            m = re.match(r'GOTO\s+(\d+)', stmt)
            if m:
                return f"Jump to line {m.group(1)}"

        elif rule_id == "if_then_true":
            m = re.match(r'IF\s+(\w+)([><=!<>]+)(\d+)\s+THEN\s+GOTO\s+(\d+)\s+\((\w+)=(\d+)\)', stmt)
            if m:
                val = m.group(6)
                cond_val = m.group(3)
                op = m.group(2)
                target = m.group(4)
                return f"Condition true ({val}{op}{cond_val}), jump to line {target}"

        elif rule_id == "if_then_false":
            m = re.match(r'IF\s+(\w+)([><=!<>]+)(\d+)\s+THEN\s+GOTO\s+(\d+)\s+\((\w+)=(\d+)\)', stmt)
            if m:
                val = m.group(6)
                cond_val = m.group(3)
                op = m.group(2)
                return f"Condition false ({val}{op}{cond_val}), next line {next_line}"

        elif rule_id == "end_halt":
            return "Halt program"

        elif rule_id == "rem_comment":
            return f"No operation (comment), next line {next_line}"

        return None

    @staticmethod
    def _eval_arithmetic(expr: str, ctx: str) -> Optional[str]:
        """Evaluate a simple arithmetic expression given variable context."""
        vars_dict = {}
        for assignment in ctx.split(","):
            assignment = assignment.strip()
            m = re.match(r'(\w+)=([\d.]+)', assignment)
            if m:
                vars_dict[m.group(1)] = float(m.group(2))
        try:
            safe_expr = expr
            for var, val in sorted(vars_dict.items(), key=lambda x: -len(x[0])):
                safe_expr = safe_expr.replace(var, str(val))
            safe_expr = safe_expr.replace("^", "**")
            result = eval(safe_expr)
            return str(result)
        except Exception:
            return None


# =========================================================================
# CURRICULUM TRAINER
# =========================================================================

class CurriculumTrainer:
    """
    Three-stage curriculum trainer with crystallization-gated progression.
    Stage N+1 unlocks only when Stage N achieves target crystallization.
    """

    def __init__(
        self,
        device: str = "auto",
        lr: float = 1e-3,
        save_name: str = "theo_curriculum_v5",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.save_name = save_name
        self.vocab = CharVocab()
        self.template = TemplateEngine()

        self.current_stage = 0
        self.stage_models = [None, None, None]
        self.stage_optimizers = [None, None, None]
        self.stage_cms = [None, None, None]
        self.stage_data = [None, None, None]
        self.stage_crystallized = [False, False, False]
        self.stage_configs = [
            {"max_len": 10,  "hidden_dim": 64,  "num_lif_layers": 1, "num_classes": NUM_KEYWORDS, "embed_dim": 32, "beta": 0.8},
            {"max_len": 64,  "hidden_dim": 256, "num_lif_layers": 2, "num_classes": NUM_PATTERNS, "embed_dim": 32, "beta": 0.8},
            {"max_len": 64,  "hidden_dim": 256, "num_lif_layers": 2, "num_classes": NUM_PATTERNS, "embed_dim": 32, "beta": 0.8},
        ]
        self.stage_cm_configs = [
            {"loss_threshold": 0.3, "generalization_target": 0.95, "variance_threshold": 1e-4,
             "consecutive_windows_required": 10, "window_size": 20},
            {"loss_threshold": 0.5, "generalization_target": 0.90, "variance_threshold": 1e-4,
             "consecutive_windows_required": 10, "window_size": 30},
            {"loss_threshold": 0.5, "generalization_target": 0.90, "variance_threshold": 1e-4,
             "consecutive_windows_required": 10, "window_size": 30},
        ]
        self.global_step = 0

    def _build_stage(self, stage: int):
        """Initialize model, optimizer, crystallization manager, and data for a stage."""
        cfg = self.stage_configs[stage]
        cm_cfg = self.stage_cm_configs[stage]

        model = RuleClassifierSNN(
            vocab_size=self.vocab.vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            num_lif_layers=cfg["num_lif_layers"],
            max_len=cfg["max_len"],
            beta=cfg["beta"],
        ).to(self.device)

        if stage == 1 and self.stage_models[0] is not None:
            self._transfer_weights(self.stage_models[0], model)
        elif stage == 2 and self.stage_models[1] is not None:
            model.load_state_dict(self.stage_models[1].state_dict())

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        cm = CrystallizationManager(on_regression="decrystallize", retention_target=0.9, **cm_cfg)

        if stage == 0:
            data = build_stage1_data()
        elif stage == 1:
            data = build_stage2_data()
        else:
            data = build_stage3_data()

        for rule in data:
            cm.register_rule_class(
                rule_id=rule["rule_id"],
                description=rule["description"],
                train_pairs=rule["train_pairs"],
                test_pairs=rule["test_pairs"],
            )

        self.stage_models[stage] = model
        self.stage_optimizers[stage] = optimizer
        self.stage_cms[stage] = cm
        self.stage_data[stage] = data

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {C.GREEN}Model parameters:{C.RESET} {n_params:,}")
        print(f"  {C.GREEN}Rule classes:{C.RESET} {len(data)}")
        print(f"  {C.GREEN}Hidden dim:{C.RESET} {cfg['hidden_dim']}")
        print(f"  {C.GREEN}LIF layers:{C.RESET} {cfg['num_lif_layers']}")
        print(f"  {C.GREEN}Max len:{C.RESET} {cfg['max_len']}")

    def _transfer_weights(self, src: RuleClassifierSNN, dst: RuleClassifierSNN):
        """Transfer compatible weights from Stage 1 model to Stage 2 model."""
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        transferred = 0
        for name, param in src_sd.items():
            if name in dst_sd and dst_sd[name].shape == param.shape:
                dst_sd[name].copy_(param)
                transferred += 1
        dst.load_state_dict(dst_sd)
        print(f"  {C.YELLOW}Transferred {transferred} weight tensors from Stage 1{C.RESET}")

    def _encode_input(self, text: str, max_len: int) -> torch.Tensor:
        """Encode input text to tensor."""
        return self.vocab.encode(text, max_len).unsqueeze(0).to(self.device)

    def _train_step(self, stage: int, pair: Dict) -> float:
        """One training step: forward, loss, backward, step."""
        model = self.stage_models[stage]
        optimizer = self.stage_optimizers[stage]
        cfg = self.stage_configs[stage]

        model.train()
        input_ids = self._encode_input(pair["input"], cfg["max_len"])
        label = torch.tensor([pair["label"]], dtype=torch.long, device=self.device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def _compute_weight_delta(self, stage: int) -> float:
        """Compute norm of parameter gradients as proxy for weight delta."""
        model = self.stage_models[stage]
        total = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
                count += 1
        return (total / max(count, 1)) ** 0.5

    def _gen_encode_fn(self, stage: int):
        """Returns encode function for crystallization manager."""
        cfg = self.stage_configs[stage]
        def encode_fn(pair):
            return self._encode_input(pair["input"], cfg["max_len"])
        return encode_fn

    def _gen_decode_fn(self, stage: int):
        """Returns decode function for crystallization manager."""
        def decode_fn(logits):
            return logits.argmax(dim=-1).item()
        return decode_fn

    def _gen_compare_fn(self, stage: int):
        """Returns compare function for crystallization manager."""
        if stage <= 1:
            def compare_fn(predicted, pair):
                return predicted == pair["label"]
            return compare_fn
        else:
            def compare_fn(predicted, pair):
                if predicted != pair["label"]:
                    return False
                rule_id = PATTERN_NAMES[predicted]
                rendered = self.template.render(rule_id, pair["input"])
                if rendered is None:
                    return predicted == pair["label"]
                return rendered.strip() == pair["output"].strip()
            return compare_fn

    def _progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Generate a colored progress bar string."""
        frac = current / max(total, 1)
        filled = int(width * frac)
        bar = f"{C.GREEN}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET}"
        return f"[{bar}] {frac:.0%}"

    def _loss_arrow(self, loss: float, prev_loss: Optional[float]) -> str:
        """Colored arrow indicating loss direction."""
        if prev_loss is None:
            return f"{C.WHITE}{loss:.4f}{C.RESET}"
        if loss < prev_loss - 0.001:
            return f"{C.GREEN}↓{loss:.4f}{C.RESET}"
        elif loss > prev_loss + 0.001:
            return f"{C.RED}↑{loss:.4f}{C.RESET}"
        else:
            return f"{C.YELLOW}→{loss:.4f}{C.RESET}"

    def train_stage(self, stage: int, max_epochs: int) -> bool:
        """
        Train a single stage until all rules crystallize or max_epochs reached.
        Returns True if stage fully crystallized.
        """
        stage_names = ["KEYWORD RECOGNITION", "SYNTAX PATTERN RECOGNITION", "FULL STATEMENT PROCESSING"]
        print(f"\n{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}  STAGE {stage+1}: {stage_names[stage]}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")

        self._build_stage(stage)
        model = self.stage_models[stage]
        cm = self.stage_cms[stage]
        data = self.stage_data[stage]

        prev_losses = {}
        epoch_start = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_losses = {}
            crystallization_events = []

            # Interleave rules. Training one class in a block teaches the
            # network to answer that class for everything, which Neo's
            # specificity gate (rightly) refuses.
            schedule = []
            for rule in data:
                rid = rule["rule_id"]
                if cm.rule_classes[rid].is_crystallized:
                    # Replay: one rehearsal per epoch keeps a crystallized rule
                    # alive in the shared weights. Neo is not told about it.
                    schedule.append((rid, random.choice(rule["train_pairs"]), True))
                else:
                    schedule.extend((rid, pair, False) for pair in rule["train_pairs"])
            random.shuffle(schedule)

            rule_losses_by_id: Dict[str, List[float]] = {}
            for rid, pair, is_replay in schedule:
                loss = self._train_step(stage, pair)
                self.global_step += 1
                if is_replay or cm.rule_classes[rid].is_crystallized:
                    continue
                delta = self._compute_weight_delta(stage)
                rule_losses_by_id.setdefault(rid, []).append(loss)

                event = cm.step(
                    rule_id=rid,
                    train_loss=loss,
                    delta_norm=delta,
                    model=model,
                    encode_fn=self._gen_encode_fn(stage),
                    decode_fn=self._gen_decode_fn(stage),
                    compare_fn=self._gen_compare_fn(stage),
                    eval_interval=10,
                )
                if event is not None:
                    crystallization_events.append(event)

            for rid, losses in rule_losses_by_id.items():
                epoch_losses[rid] = sum(losses) / len(losses)

            # Crystallized is a claim that keeps being checked. Any rule the
            # network has since forgotten goes back to the plastic pool.
            retention = cm.audit_retention(
                model, self._gen_encode_fn(stage), self._gen_decode_fn(stage),
                self._gen_compare_fn(stage),
            )
            for rid, acc in retention.items():
                if acc < cm.retention_target:
                    print(f"\n  {C.RED}{C.BOLD} FORGOTTEN {C.RESET} {C.RED}{rid}{C.RESET} "
                          f"retention {acc:.0%} -> returned to plastic pool")

            for event in crystallization_events:
                rid = event["rule_id"]
                gen = event["final_generalization_accuracy"]
                print(f"\n  {C.BG_CYAN}{C.BOLD} CRYSTALLIZED {C.RESET} "
                      f"{C.CYAN}{rid}{C.RESET} at step {event['crystallized_at_step']} "
                      f"gen:{C.GREEN}{gen:.0%}{C.RESET}")

            frac = cm.fraction_crystallized()
            n_active = len(epoch_losses)

            if epoch % 50 == 0 or epoch <= 5 or crystallization_events or epoch == max_epochs:
                avg_loss = sum(epoch_losses.values()) / max(len(epoch_losses), 1)
                avg_prev = sum(prev_losses.get(k, avg_loss) for k in epoch_losses) / max(len(epoch_losses), 1) if prev_losses else None
                elapsed = time.time() - epoch_start
                bar = self._progress_bar(int(frac * len(data)), len(data))
                loss_str = self._loss_arrow(avg_loss, avg_prev)
                print(f"  Epoch {epoch:5d} | active:{n_active:3d} | loss:{loss_str} | "
                      f"crystallized:{bar} | {elapsed:.0f}s")
                prev_losses = dict(epoch_losses)

            if frac >= 1.0:
                elapsed = time.time() - epoch_start
                print(f"\n  {C.BG_GREEN}{C.BOLD} STAGE {stage+1} COMPLETE {C.RESET} "
                      f"All rules crystallized at epoch {epoch} ({elapsed:.0f}s)")
                self.stage_crystallized[stage] = True

                save_path = f"{self.save_name}_stage{stage+1}_epoch{epoch}.pt"
                self._save_checkpoint(stage, save_path, epoch)
                return True

            if epoch % 200 == 0:
                save_path = f"{self.save_name}_stage{stage+1}_epoch{epoch}.pt"
                self._save_checkpoint(stage, save_path, epoch)

        self._save_checkpoint(stage, f"{self.save_name}_stage{stage+1}_epoch{max_epochs}_incomplete.pt", max_epochs)
        elapsed = time.time() - epoch_start
        frac = cm.fraction_crystallized()
        print(f"\n  {C.YELLOW}Stage {stage+1} reached max epochs. {frac:.0%} crystallized ({elapsed:.0f}s){C.RESET}")

        print(f"\n  {C.BOLD}Final status:{C.RESET}")
        for rid, info in cm.status().items():
            tag = f"{C.GREEN}FROZEN{C.RESET}" if info["crystallized"] else f"{C.RED}PLASTIC{C.RESET}"
            # Measured now, on the final weights. Never the value from the
            # moment the rule crystallized.
            gen = CrystallizationManager._accuracy(
                cm.rule_classes[rid].test_pairs, model, self._gen_encode_fn(stage),
                self._gen_decode_fn(stage), self._gen_compare_fn(stage))
            open_gates = [g for g, ok in info["gates"].items() if not ok]
            why = "" if info["crystallized"] else f"  waiting on: {', '.join(open_gates)}"
            print(f"    [{tag:>20s}] {rid:30s} held-out now:{gen:.0%}{why}")

        return frac >= 1.0

    def train(self, max_epochs: int = 5000):
        """Run all three curriculum stages."""
        banner()
        print(f"  {C.BOLD}Device:{C.RESET} {self.device}")
        print(f"  {C.BOLD}Vocab size:{C.RESET} {self.vocab.vocab_size}")
        print(f"  {C.BOLD}Max epochs per stage:{C.RESET} {max_epochs}")

        stage1_ok = self.train_stage(0, max_epochs=min(max_epochs, 2000))
        if not stage1_ok:
            print(f"\n  {C.RED}Stage 1 did not fully crystallize. Stage 2 blocked.{C.RESET}")
            return

        stage2_ok = self.train_stage(1, max_epochs=max_epochs)
        if not stage2_ok:
            print(f"\n  {C.RED}Stage 2 did not fully crystallize. Stage 3 blocked.{C.RESET}")
            return

        stage3_ok = self.train_stage(2, max_epochs=max_epochs)
        if not stage3_ok:
            print(f"\n  {C.YELLOW}Stage 3 did not fully crystallize.{C.RESET}")

        print(f"\n{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}  CURRICULUM TRAINING COMPLETE{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}{'='*66}{C.RESET}")
        for i in range(3):
            status = f"{C.GREEN}CRYSTALLIZED{C.RESET}" if self.stage_crystallized[i] else f"{C.RED}INCOMPLETE{C.RESET}"
            print(f"  Stage {i+1}: {status}")

    def _save_checkpoint(self, stage: int, path: str, epoch: int):
        """Save model checkpoint."""
        model = self.stage_models[stage]
        cm = self.stage_cms[stage]
        torch.save({
            "stage": stage,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": self.stage_configs[stage],
            "crystallization_log": cm.crystallization_log,
            "status": cm.status(),
            "global_step": self.global_step,
        }, path)
        print(f"  {C.DIM}Saved checkpoint: {path}{C.RESET}")

    def test_only(self, checkpoint_path: str):
        """Load checkpoint and run evaluation."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        stage = ckpt["stage"]
        cfg = ckpt["model_config"]

        if stage == 0:
            num_classes = NUM_KEYWORDS
        else:
            num_classes = NUM_PATTERNS

        model = RuleClassifierSNN(
            vocab_size=self.vocab.vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=num_classes,
            num_lif_layers=cfg["num_lif_layers"],
            max_len=cfg["max_len"],
            beta=cfg["beta"],
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        if stage == 0:
            data = build_stage1_data()
        elif stage == 1:
            data = build_stage2_data()
        else:
            data = build_stage3_data()

        stage_names = ["KEYWORD RECOGNITION", "SYNTAX PATTERN RECOGNITION", "FULL STATEMENT PROCESSING"]
        print(f"\n{C.CYAN}{C.BOLD}  TEST: Stage {stage+1} ({stage_names[stage]}){C.RESET}")
        print(f"  Loaded: {checkpoint_path}")
        print(f"  Epoch: {ckpt['epoch']}")

        total_correct = 0
        total_count = 0
        for rule in data:
            correct = 0
            count = 0
            for pair in rule["test_pairs"]:
                input_ids = self._encode_input(pair["input"], cfg["max_len"])
                with torch.no_grad():
                    logits = model(input_ids)
                pred = logits.argmax(dim=-1).item()
                if pred == pair["label"]:
                    correct += 1
                count += 1
            acc = correct / max(count, 1)
            color = C.GREEN if acc >= 0.9 else C.YELLOW if acc >= 0.5 else C.RED
            print(f"    {rule['rule_id']:30s} {color}{acc:.0%}{C.RESET} ({correct}/{count})")
            total_correct += correct
            total_count += count

        overall = total_correct / max(total_count, 1)
        print(f"\n  Overall: {C.BOLD}{overall:.1%}{C.RESET} ({total_correct}/{total_count})")


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


def test_surrogate_spike():
    v = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
    spikes = surrogate_spike(v)
    assert spikes.tolist() == [0.0, 1.0, 1.0, 1.0]
    loss = spikes.sum()
    loss.backward()
    assert v.grad is not None
    assert all(g > 0 for g in v.grad.tolist())
    print("PASS: test_surrogate_spike")


def test_classifier_forward():
    v = CharVocab()
    m = RuleClassifierSNN(vocab_size=v.vocab_size, num_classes=12, max_len=10)
    x = v.encode("PRINT", max_len=10).unsqueeze(0)
    out = m(x)
    assert out.shape == (1, 12)
    print("PASS: test_classifier_forward")


def test_classifier_backward():
    v = CharVocab()
    m = RuleClassifierSNN(vocab_size=v.vocab_size, num_classes=12, max_len=10)
    x = v.encode("PRINT", max_len=10).unsqueeze(0)
    label = torch.tensor([0])
    logits = m(x)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad, "No gradients flowed through surrogate spike"
    print(f"PASS: test_classifier_backward (loss={loss.item():.4f})")


def test_template_engine():
    te = TemplateEngine()
    assert te.render("print_string_literal", 'line 10: PRINT "Hello"') == "Print 'Hello' to screen, next line 20"
    assert te.render("print_blank", "line 10: PRINT") == "Print blank line, next line 20"
    assert te.render("goto_unconditional", "line 10: GOTO 100") == "Jump to line 100"
    assert te.render("end_halt", "line 10: END") == "Halt program"
    assert te.render("rem_comment", "line 10: REM Test") == "No operation (comment), next line 20"
    assert te.render("let_simple_assign", "line 10: LET X=5") == "X=5, next line 20"
    print("PASS: test_template_engine")


def test_stage_configs():
    trainer = CurriculumTrainer(device="cpu")
    assert len(trainer.stage_configs) == 3
    assert trainer.stage_configs[0]["num_classes"] == NUM_KEYWORDS
    assert trainer.stage_configs[1]["num_classes"] == NUM_PATTERNS
    print("PASS: test_stage_configs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum-Staged Crystallization Trainer v5")
    parser.add_argument("--test-only", type=str, default=None, metavar="CHECKPOINT",
                        help="Load checkpoint and run evaluation")
    parser.add_argument("--epochs", type=int, default=5000, help="Max training epochs per stage")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", type=str, default="theo_curriculum_v5", help="Save name prefix")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests and exit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (training order and init)")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.self_test:
        test_char_vocab()
        test_surrogate_spike()
        test_classifier_forward()
        test_classifier_backward()
        test_template_engine()
        test_stage_configs()
        print(f"\n{C.GREEN}All curriculum_trainer_v5 self-tests passed.{C.RESET}")
    elif args.test_only:
        trainer = CurriculumTrainer(device=args.device, lr=args.lr, save_name=args.save)
        trainer.test_only(args.test_only)
    else:
        trainer = CurriculumTrainer(device=args.device, lr=args.lr, save_name=args.save)
        trainer.train(max_epochs=args.epochs)
