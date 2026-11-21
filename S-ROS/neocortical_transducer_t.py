"""
neocortical_transducer_t.py
Neocortical Translation Layer: Spike-to-Continuous Population Decoding

BIOLOGICAL GROUNDING
This file models the association cortices where diverse cortical streams 
converge. It acts as the translation layer bridging discrete machine-state 
logic and continuous semantic memory before projecting to the entorhinal cortex.
It unifies the dorsal action stream and the ventral meaning stream into a 
singular representation.
Mesulam MM (1998). "From sensation to cognition." Brain, 121(6), 1013-1052. 
DOI: 10.1093/brain/121.6.1013
"""
import torch
import torch.nn as nn

class NeocorticalTransducer(nn.Module):
    """
    Association cortex convergence layer for multimodal integration.
    BIOLOGICAL STRUCTURE: Multimodal Association Cortex.
    BIOLOGICAL FUNCTION: Binds parallel, segregated processing streams 
    into unified, higher-order representations.
    Mesulam MM (1998). "From sensation to cognition." Brain, 121(6), 1013-1052. 
    DOI: 10.1093/brain/121.6.1013
    ANATOMICAL INTERFACE:
    Sending structures: Dorsal stream regions and Ventral stream regions.
    Receiving structure: Entorhinal cortex layer II/III.
    Connection: Parahippocampal projections.
    """
    def __init__(self, spike_dim: int, coordinate_dim: int = 64) -> None:
        super().__init__()
        self.optimal_linear_decoder = nn.Linear(spike_dim, coordinate_dim, bias=False)
        self.semantic_projection = nn.Linear(coordinate_dim, coordinate_dim, bias=True)
        self.unification_layer = nn.Linear(coordinate_dim * 2, coordinate_dim, bias=True)

    def decode_population_spikes(self, discrete_spikes: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete orthogonal spikes to a continuous vector space.
        BIOLOGICAL STRUCTURE: Cortical Population Coding.
        BIOLOGICAL FUNCTION: Extracts continuous variables from the 
        collective activity of discrete spiking neurons.
        Eliasmith C, Anderson CH (2003). "Neural engineering: Computation, 
        representation, and dynamics in neurobiological systems." MIT Press. 
        DOI: 10.7551/mitpress/4934.001.0001
        NOT a biological quantity: The linear weights are trained artifacts 
        acting as decoding filters, not direct synaptic weights.
        """
        return self.optimal_linear_decoder(discrete_spikes)

    def forward(self, dorsal_spikes: torch.Tensor, ventral_floats: torch.Tensor) -> torch.Tensor:
        """
        Concatenate and project streams into a shared coordinate space.
        BIOLOGICAL STRUCTURE: Neocortical mixture of experts binding.
        BIOLOGICAL FUNCTION: Integrates inputs from distinct specialized 
        networks into a singular manifold for hippocampal processing.
        Fuster JM (2001). "The prefrontal cortex--an update." 
        Neuron, 30(2), 319-333. DOI: 10.1016/S0896-6273(01)00285-9
        """
        decoded_state = self.decode_population_spikes(dorsal_spikes)
        semantic_state = torch.tanh(self.semantic_projection(ventral_floats))
        unified = torch.cat([decoded_state, semantic_state], dim=-1)
        return torch.tanh(self.unification_layer(unified))
      
