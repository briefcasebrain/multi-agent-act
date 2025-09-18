"""
Knowledge distillation and transfer mechanisms for multi-agent learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class KnowledgeDistillationEngine(nn.Module):
    """
    Engine for knowledge distillation between agents.

    This module provides functionality for compressing, transferring, and
    decompressing knowledge between teacher and student agents, including
    quality assessment and cross-agent knowledge alignment.
    """

    def __init__(self, feature_dim: int = 512, distillation_temperature: float = 4.0):
        """
        Initialize the knowledge distillation engine.

        Args:
            feature_dim: Dimensionality of feature representations
            distillation_temperature: Temperature parameter for softmax in distillation
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.temperature = distillation_temperature

        # Knowledge compression network
        self.knowledge_compressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim // 8)
        )

        # Knowledge decompression network
        self.knowledge_decompressor = nn.Sequential(
            nn.Linear(feature_dim // 8, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

        # Knowledge quality assessor
        self.quality_assessor = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # [original, reconstructed]
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Cross-agent knowledge alignment
        self.alignment_network = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

    def compress_knowledge(self, teacher_knowledge: torch.Tensor) -> torch.Tensor:
        """
        Compress teacher knowledge for efficient transfer.

        Args:
            teacher_knowledge: High-dimensional knowledge representation

        Returns:
            Compressed knowledge representation
        """
        compressed = self.knowledge_compressor(teacher_knowledge)
        return compressed

    def decompress_knowledge(self, compressed_knowledge: torch.Tensor) -> torch.Tensor:
        """
        Decompress knowledge for student agent.

        Args:
            compressed_knowledge: Compressed knowledge representation

        Returns:
            Decompressed knowledge representation
        """
        decompressed = self.knowledge_decompressor(compressed_knowledge)
        return decompressed

    def distill_knowledge(self, teacher_outputs: torch.Tensor,
                         student_outputs: torch.Tensor) -> torch.Tensor:
        """
        Perform knowledge distillation between teacher and student.

        Args:
            teacher_outputs: Teacher model outputs (logits)
            student_outputs: Student model outputs (logits)

        Returns:
            Knowledge distillation loss
        """
        # Soften predictions with temperature
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)

        # KL divergence loss
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)

        return distillation_loss

    def assess_knowledge_quality(self, original_knowledge: torch.Tensor,
                                transferred_knowledge: torch.Tensor) -> torch.Tensor:
        """
        Assess quality of knowledge transfer.

        Args:
            original_knowledge: Original knowledge representation
            transferred_knowledge: Transferred knowledge representation

        Returns:
            Quality score between 0 and 1
        """
        combined = torch.cat([original_knowledge, transferred_knowledge], dim=1)
        quality_score = self.quality_assessor(combined)

        return quality_score

    def align_cross_agent_knowledge(self, agent_knowledge_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Align knowledge representations across multiple agents.

        Args:
            agent_knowledge_list: List of knowledge representations from different agents

        Returns:
            Aligned knowledge representation
        """
        if len(agent_knowledge_list) < 2:
            return agent_knowledge_list[0] if agent_knowledge_list else torch.zeros(1, self.feature_dim)

        # Stack knowledge representations
        knowledge_batch = torch.stack(agent_knowledge_list)

        # Apply cross-attention for alignment
        aligned_knowledge, attention_weights = self.alignment_network(
            knowledge_batch, knowledge_batch, knowledge_batch
        )

        # Return averaged aligned knowledge
        return torch.mean(aligned_knowledge, dim=0)