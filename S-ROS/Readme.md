# S-ROS (Spiking Robot Operating System)
## Project Goals

1) Prove that a de novo spiking neural network can reach a transition prediction loss of < 0.007 (0.7% MSE) 

2) Prove that a de novo spiking neural network can learn procedural, syntactic code execution (BASIC-style logic) without pre-training or distillation.

3) Establish a biologically grounded neuro-symbolic bridge that completely isolates exploratory learning from rigid, near zero-hallucination robotic actuation laying the ground work for learning systems which can reach an Interal Crystallization Threshold of 0.7% MSE while also containing one shot learning mechanics

4) Prove that a de novo spiking neural network can overcome the limits of standard Finite State Machines by implementing a discrete Spiking Call Stack, allowing the network to natively track GOSUB recursion and nested loops without state aliasing.
  



#### Chatbots are a solved problem, but standard robotic operating systems are rigid message-passers that cannot adapt. S-ROS marries the absolute, deterministic certainty of procedural logic (BASIC/ROS) with the associative, real-time learning capabilities of a pure spiking neural network.

## A system that predicts the next token and then stores memories of its own predictions is a system that just remembers what it guessed. 
#### It does not understand what it guessed or why. Understanding requires the system to build an internal model where the relationships between elements are represented explicitly enough that the system can answer questions it was never trained on by composing what it knows. When a human learns that PRINT with a comma puts output in the next zone, they do not just memorize the input-output pair. They build a model of what zones are, what commas do as separators, and what PRINT does as a display operation. Those three pieces compose freely: they can predict what PRINT A$, B$, C$ does without ever having seen a three-argument example, because they understand the comma rule, not just the specific case.

# Learning Order

Before the comma rule can be learned, the system must first learn and crystallize every individual key press. There is no shortcut. The comma rule is a composition over keys. If the keys themselves are not frozen engrams, there is nothing to compose over.

This is the explicit staged curriculum. Each stage gates on crystallization of the previous one. No stage begins until the prior stage has frozen everything it was asked to freeze.

## Stage Zero: Motor Babbling

The system learns that output spike position N corresponds to key N. For every key in the set {A-Z, 0-9, SPACE, COMMA, ENTER}, which is 39 keys total, Theo trains on the identity mapping (pressing key X produces the spike at position X) until Neo says the pathway generalizes. At that point the (sensor, action) pair is written into Theo's S-CAM as a frozen engram.

Crystallization criteria for each key:

1. Training loss below threshold
2. Held-out generalization at 100% across 3 consecutive evaluation windows
3. Weight delta variance stabilized

All three must hold simultaneously before the key is committed to S-CAM.

Keys are trained sequentially, one at a time. A earns its engram first. Then B. Then C. And so on through ENTER. Plastic weights drift as later keys train, but the S-CAM engrams are frozen data, not gradient-reachable, so keys that crystallized earlier cannot be forgotten. Inference order is S-CAM exact match first, trainable executor second. A matching engram always wins over a drifting executor.

Stage Zero is not complete until all 39 keys have crystallized. If a key fails to crystallize within the step budget, the run has failed and must be debugged before proceeding. No subsequent stage can run on a system missing any of its base keys.

## Stage One: Single Character On Request

Prerequisite: Stage Zero complete. All 39 keys in S-CAM.

The system learns that a request encoding (not the key itself) should produce the matching key. Stage Zero trained "seeing A produces A." Stage One trains "being asked for A produces A." This is a different sensor vector mapping to the same action vector. Stage Zero's engrams are preserved because the sensor patterns are distinct from Stage One's request patterns, so S-CAM retrievals do not collide.

## Stage Two: Next Key In Sequence

Prerequisite: Stage One complete.

The system learns that given key X as a cue, the next key in the alphabet or sequence should fire. This is the first stage that requires Cognitive Kernel CA3 attractor completion rather than S-CAM exact match. The sensor is not the target. The sensor is a cue that the kernel completes into the target.

## Stage Three: Short Sequences

Prerequisite: Stage Two complete.

The system learns that it can type a fixed short sequence (like "HI" or "42") on request. This is the first stage that requires CA1 and Subiculum working memory to hold the current position in the output sequence.

## Stage Four: The Comma Rule

Prerequisite: Stages Zero through Three complete.

The system learns the compositional PRINT-comma rule, which says that a comma between PRINT arguments produces zone-separated output. This is the target demonstration of the whole system. It is not asked of the system until every prerequisite has crystallized.

The comma rule is compositional. It combines the key engrams from Stage Zero, the request-to-key mapping from Stage One, the next-key completion from Stage Two, and the sequence-holding from Stage Three. If any one of those is missing or unreliable, the comma rule cannot be learned. It can only be memorized, which Neo's held-out generalization gate will catch and refuse to crystallize.

## Why The Order Matters

PRAGMI's claim is that the system learns rules, not specific examples. The only way to demonstrate that claim is to verify each compositional primitive in isolation before asking the system to compose them.

A system that learns the comma rule without first individually crystallizing every key could be memorizing comma-rule training examples at the character level. A system that first crystallized all 39 keys, then all 39 requests, then sequence completion, then sequence holding, and only then attempts the comma rule, can be observed to generalize. Neo's gate will have already refused to crystallize anything that looked like memorization at every prior stage.

This is what makes the generalization claim falsifiable instead of asserted. Every stage is a place the system could fail, and the failure would be visible before it propagated into the next stage.

## Constraint Compliance

Every stage is trained under the same six architectural constraints:

1. Crystallization manager observes only keyboard and screen
2. Action decoder is a fixed lookup, hashed, verified on every load
3. Every probe is shown individually
4. Real C64ScreenBuffer
5. Keyboard is the only input path to Theo
6. No loss visible. MATCH/MISS and screen only.

Adding stages does not relax these constraints. Each new stage must pass the preflight `selftest.py` before it begins.

# The real learning begins 

## Stage 1: Keyword Recognition Complete
<img width="1212" height="1333" alt="S-ROSS Training" src="https://github.com/user-attachments/assets/e0213736-a640-4d50-8487-fbca222e1471" />

![Stage 1 Crystallization - All keywords locked in](./images/stage1-complete.png)
Stage 1 successfully crystallized 12 keywords with 100% generalization accuracy.


## Rote Learning + Crystallization Pipeline

**rote_data_generator.py**  
Generates 15 BASIC rule classes and splits data into Training Pairs and Held-Out Test Pairs.

**hello_world_trainer.py**  
1. Encodes BASIC lines  
2. Runs RoteLearner (LIF Network)  
3. Computes cross-entropy loss  
4. Backpropagates & updates weights  

After every N training steps, it hands control to:

**crystallization_manager.py (Neo)**  
Tests the network on the Held-Out Test Pairs.  
Checks 3 conditions for K consecutive windows:  
• Training loss < threshold  
• Weight delta variance stabilized  
• Generalization accuracy > target  

If all 3 pass → **Rule Crystallized!** (skipped in future epochs)  
If not → continue training next epoch.

## Current state (repaired November 2026)

What is proven, and by which command:

| Claim | Command | Result |
|---|---|---|
| Neo refuses a lookup table, a constant predictor and a steady gradient, and accepts a genuine rule | `python3 crystallization_manager.py` | all tests pass |
| The S-CAM is exact, write-once, never evicts, and is not reachable by gradient | `python3 theo_scam.py` | 9/9 pass |
| A frozen engram wins over the executor and survives executor training | `python3 Theo_Core_t.py` | 5/5 pass |
| **Stage Zero: 39 keys, each gated by Neo, written to the S-CAM, kept after the executor is deliberately un-taught, and restored from a `.soul` file** | `python3 stage_zero_keys.py` | passes (about 75 s on one CPU core) |
| Curriculum stage 1 (12 keywords) crystallizes under the strict gates | `python3 curriculum_trainer_v5.py --epochs 120 --device cpu` | 12/12 in 28 epochs |

What is not proven:

- Curriculum stages 2 and 3 have not been run to completion since the repair (about 25 s per epoch on CPU; use a GPU).
- `hello_world_trainer.py` cannot crystallize its rules: its model has no way to copy an unseen literal. See the note in `RoteLearner`.
- Stages One to Four of the Learning Order below are not implemented. Only Stage Zero is.
- Of the six Constraint Compliance rules, only "action decoder is a fixed lookup" and "every probe is shown individually" hold in `stage_zero_keys.py`. There is no `selftest.py`, and no trainer uses `C64ScreenBuffer`.
- `c64_basic_trace_generator.py` does not implement PRINT comma zones or semicolons, and `c64_basic_traces.jsonl` is not read by any trainer.

Set `OMP_NUM_THREADS=1` on CPU: these models are small and multi-threading makes each step about ten times slower.

## Usage
### python3.11 -u curriculum_trainer_v5.py --epochs 5000 --device cuda --save theo_curriculum_v5



# What the files are supposed to do 
## reward_modulator_t.py
is the implementation of the dopamine reward prediction error (RPE) module that gives Tiny Tim a fast credit-assignment loop on top of its slower STDP substrate. It models the mesolimbic dopamine system, standing in for the ventral tegmental area's projection to striatum and prefrontal cortex, where dopaminergic neurons encode the signed difference between received and predicted reward and broadcast that scalar diffusely to modulate concurrent plasticity. Each game tick, the module receives Tiny Tim's predicted outcome and the actual reward from Theo's spike bus output, computes the RPE scalar d, and gates the STDP weight update as delta_w * (1 + d * eligibility_trace) before commit. The eligibility trace implements synaptic tagging, bridging the temporal gap between action and delayed reward by accumulating recent weight-change magnitudes with exponential decay, so only recently active synapses receive the full modulation when reward arrives. Five ablation flags independently disable each mechanism (master reward modulation, prediction subtraction, eligibility trace, dopamine saturation, baseline subtraction), enabling a clean factorial ablation study for the paper. The module preserves Neo's role as the crystallization substrate monitor: Neo watches the post-modulation weight deltas, which is correct because crystallization should track the modulated learning trajectory rather than raw STDP. All state is captured in registered buffers with full round-trip serialization via to_dict and from_dict.
## crystallization_manager.py
is the implementation of Neo, the crystallization monitor. Its core function is to answer one question: does the network understand a rule, or has it just memorized specific examples? It maintains a registry of rule classes, where each class has training instances the network learns from and held-out test instances the network never sees during training. After every N training steps, it runs the network on those held-out instances and checks whether the network gets them right. A rule only crystallizes when three conditions are met simultaneously for K consecutive evaluation windows: training loss is below threshold, weight delta variance has stabilized, and generalization accuracy on unseen instances exceeds the target. That third condition is what makes this different from every other convergence monitor. Without it, you get a system that memorizes. With it, you get a system that learns. The module is deliberately standalone with no dependencies on the Cognitive Kernel so it can be tested in isolation. Once it works here with Theo on rote BASIC data, it integrates into the kernel without rewriting.
## rote_data_generator.py
produces the Option B training data organized by compositional rule. There are 15 rule classes covering PRINT variants, LET variants, GOTO, IF/THEN true and false branches, END, and REM. Each class contains training pairs and held-out test pairs that exercise the same rule with different specific values. The held-out pairs are guaranteed to never appear in the training set. This structure is what lets the crystallization manager distinguish memorization from understanding: if the network can handle PRINT "Cat","Dog" after only training on PRINT "Hello","world" and PRINT "A","B", it learned the comma-means-zones rule, not three specific strings.
## hello_world_trainer.py
is the training loop that wires the other two together with a minimal LIF recurrent network called RoteLearner. It encodes BASIC lines and their semantic descriptions as character sequences, feeds them through a 2-layer spiking recurrent network with LIF neuron dynamics, computes cross-entropy loss against the expected output, and backpropagates. After training steps, it hands control to the crystallization manager which evaluates generalization on held-out pairs and decides whether to freeze. The loop iterates over all non-crystallized rule classes each epoch, skipping any rule that has already been frozen. Training ends when all 15 rules crystallize or max epochs is reached. The output is a checkpoint containing the model weights, the crystallization log, and the full status of every rule class. This is the smallest possible closed loop that proves or disproves whether an SNN can learn compositional rules through rote exposure and crystallize genuine understanding.
## theo_checkpoint.py
is the engineering persistence layer for the TheoCore module, managing the saving and loading of network checkpoints as .soul files. It safely serializes the complete identity state of the organism, including the S-CAM engram buffers, the recurrent executor weights, and the crystallization ledger. It uses an atomic write pattern by writing to a temporary file before replacing the original, which ensures checkpoints are never left in a corrupted state if a save is interrupted. During the loading process, it performs strict hyperparameter validation by checking architecture dimensions like spike dimension, coordinate dimension, and S-CAM capacity. If any dimension mismatches, it immediately raises an error specifying the exact discrepancy. This strict validation guarantees architectural consistency on load, ensuring that the network's crystallized skills and memory states are perfectly preserved and accurately restored without neural dynamics corruption.
## cortical_buffer_teaching.py
is the implementation of the prefrontal cortex persistent activity mechanism, functioning as a working memory buffer that sustains network representations across time in the absence of external input. It compresses a cortical column's post-forward-pass membrane state into a compact 32-dimensional bottleneck and injects it as a sub-threshold additive bias on the initial membrane potential of the next forward pass. This effectively keeps prior context in mind to prime future sensory processing. This state evolves via an exponential moving average rather than a hard overwrite, allowing it to act as a dynamical attractor that accumulates a weighted history of recent network states. Biologically grounded in recurrent synaptic reverberation, the module dynamically adapts to structural plasticity by cleanly separating its runtime buffer state from its learned projection weights, enabling the buffer's representational width to expand without disrupting its functional memory.
## neocortex_helper_t.py
is the implementation of neocortical translation utilities that model how association cortices bridge distinct neural representations. It unifies deterministic execution memory spikes with continuous semantic float embeddings so they can be integrated for downstream processing. It accomplishes this through three core mechanisms: a population decoder that translates discrete, orthogonal spiking activity into a continuous vector space; a schema overlap metric that calculates the geometric cosine similarity between incoming memory episodes and established semantic knowledge to quantify representational overlap; and a multimodal binder that concatenates and projects these decoded action states and semantic streams into a singular, shared coordinate manifold. By mathematically translating and fusing these diverse inputs, the module prepares unified multimodal representations essential for hippocampal projection and higher-order cognitive processing.
## TheoLIF_Executor.py
is the implementation of a discrete-time Leaky Integrate-and-Fire recurrent executor designed to emulate finite state machine dynamics. It processes a retrieved one-hot memory engram into a final, stable spike train for action decoding, leveraging high noise resilience and extended state-holding periods. It achieves this by running a multi-step recurrent simulation where membrane potentials continuously integrate both the initial state and internal recurrent spikes, decaying over time according to a beta parameter, and emitting discrete spikes whenever a fixed threshold is crossed. Crucially, its recurrent and input weights are entirely frozen after crystallization from the plastic core, ensuring that once a specific functional behavior is learned, its execution remains deterministic, stable, and immune to further synaptic drift.
## c64_basic_trace_generator.py
is the ground-truth oracle and training data generator for the Theo procedural execution core. It produces deterministic execution traces that serve as the supervised learning signal for the network. It operates by implementing a custom C64 BASIC V2 subset interpreter that evaluates programs statement by statement, capturing the exact, noise-free state of the machine including the program counter, variables, loop contexts, and call stacks immediately before and after every instruction. By serializing these before-and-after snapshots into a structured JSONL format, it provides the precise narrative of state changes that the neural network's plastic side must learn to predict, functioning purely as an engineering utility with no biological neural dynamics of its own.
## world_model_ensemble_t.py (was world_model_ensemble_.py)
is the implementation of the system's generative world model, modeling the predictive coding function of neocortical circuits. It continuously predicts the network's next internal state and computes epistemic uncertainty to drive curiosity and active exploration. It runs an ensemble of five independent multi-layer perceptron prediction heads on a 64-dimensional coordinate manifold. When the heads disagree, which is measured as ensemble variance, it signals that the current region of experience is undersampled and represents high expected information gain. Uniquely within the CuriosityHead refactor, this module maintains its own learnable parameters and Adam optimizer, training on detached coordinate targets to ensure its generative updates do not contaminate the main spiking network's backpropagation graph. By quantifying prediction errors and ensemble variance, it provides the vital uncertainty metrics that downstream modules use to rank candidate data batches and compute the network's global maturity.
## c64_basic_traces.jsonl
is the serialized output from the ground-truth oracle, containing the deterministic execution traces generated by the C64 BASIC V2 subset interpreter. It provides the supervised training signal for the neural network's plastic side, consisting of structured JSON objects where each line records the complete state of the machine immediately before and after a single BASIC instruction is executed. It exhaustively details changes to the program counter, variables, loop contexts, data pointers, and output for 20 diverse test programs covering assignments, loops, arithmetic, conditionals, and subroutine jumps. By capturing these precise, noise-free transitions, this file serves as the exact narrative of state changes that the neural network must learn to predict to demonstrate procedural understanding.
## tiny_Tim.py
is the consolidated smoke-test harness for the PRAGMI architecture, combining the full Hippocampal Core with structural expansion mechanisms to verify system integrity before deployment. It executes a comprehensive self-check suite that tests four critical capabilities: forward propagation with epistemic doubt calculation, perfect save and load state recovery for working memory, dynamic structural expansion where heavily used specialist columns grow by 64 neurons, and episodic storage in the CA3 attractor network driven by novelty. It implements the entire hippocampal pathway including the Dentate Gyrus, CA3, CA1, Subiculum, Entorhinal Cortex, and an Astrocytic Regulator, and integrates them into a TinyTimmyEnsemble that simulates a multi-column working environment. This functions as the definitive green light script to ensure all biological analogs and engineering mechanisms are operating nominally.
## small_core_train_t.py
is the integration training harness for the PRAGMI system, modeling the organism-level learning cycle of experience, encoding, consolidation, and prediction refinement. It wires together five core biological subsystems (the CognitiveKernel, WorldModelEnsemble, NeuromodulatorBroadcast, EpistemicSelector, and CorticalBuffer) and orchestrates them through a simulated biological wake-sleep cycle. It continuously processes token batches during the wake phase, utilizing temporary engineering stand-ins like the LightweightEncoder and FixedCoordinateProjection to map data into a 64-dimensional coordinate manifold pending the integration of the spiking TimmyModel. Throughout the continuous training loop, it refines the generative world model, updates global neuromodulatory baselines based on prediction errors, and periodically triggers sleep consolidation for hippocampal memory replay. To ensure architectural integrity, it strictly executes an 18-point diagnostic self-test suite before training begins, validating all subsystem contracts, gradient isolations, and state serialization pipelines.
## theo_scam.py (was Theo_S-CAM_t.py)
is the implementation of a 64-Kilobyte Spiking Content-Addressable Memory module that acts as a massive procedural RAM for the deterministic core. It stores and executes complex, nested procedural logic by hybridizing associative memory lookups with sequential program counter execution. It maps up to 65,536 instructions into 128-dimensional one-hot encoded spike vectors, maintaining dedicated buffers for sensor keys, action values, and active memory states. During operation, it can either sequentially advance its internal program counter to emit the next action spike, or perform associative lookups. This involves comparing incoming sensor spikes against stored keys to execute direct jumps, GOTOs, or interrupts by snapping the program counter to a new address when a high-confidence match is found. By combining this dual-mode execution with a targeted crystallization mechanism for writing new skills, it provides the stable, large-scale procedural memory foundation required for S-ROS to reliably emulate complex finite state machines.
## cognitive_kernel_core_t.py
is the implementation of the Cognitive Kernel Learning Core, modeling the hippocampal episodic reconstruction system and its interactions with the entorhinal cortex. Its primary job is to serve as the plastic learning core of the architecture that sits beside the deterministic execution core, handling one-shot encoding, pattern separation, attractor-based pattern completion, and novelty detection. It achieves this by simulating the classical trisynaptic circuit where the Dentate Gyrus orthogonalizes inputs, CA3 stores rapidly acquired attractor traces, CA1 computes novelty by comparing reconstructions against cortical drive, and the Subiculum returns the result to the cortex. Throughout the wake-sleep cycle, it manages short-term and working memory persistence and orchestrates sleep-associated replay, ultimately reconstructing experiences from partial cues and emitting provisional reconstructions and mismatch signals upward to the neocortical mediation layer before any permanent crystallization occurs.
## Train_tiny_Tim.py
is the training script for the tiny Cognitive Kernel smoke-test. It trains the kernel on next-coordinate prediction within a 64-dimensional manifold. It runs full wake and sleep cycles alongside targeted dynamic expansion of the busiest specialist column. It achieves this by configuring the kernel and a TinyTimmyEnsemble, utilizing an AdamW optimizer to run a continuous training loop for a specified number of steps. Periodically, it triggers sleep consolidation to simulate memory replay and structural growth. Finally, it saves comprehensive checkpoints containing the model state, ensemble state, and current doubt levels, providing a verifiable log of the system's physiological learning cycle.



