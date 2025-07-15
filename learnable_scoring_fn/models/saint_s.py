"""SAINT-s (Self-Attention only) for interval prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseScoringFunction


class SAINTSScoringFunction(BaseScoringFunction):
    """SAINT-s (self-attention only variant) for interval prediction.
    
    SAINT uses separate embedding for each feature and applies
    self-attention to model inter-feature dependencies.
    """
    
    def __init__(self, input_dim: int = 17, d_embed: int = 48, n_heads: int = 3,
                 n_layers: int = 2, dropout: float = 0.1, use_mixup: bool = True,
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Number of input features
            d_embed: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            use_mixup: Whether to use mixup augmentation
                    scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.use_mixup = use_mixup
        
        # Feature-specific embedders (treating continuous features as single-value categoricals)
        self.feature_embedders = nn.ModuleList()
        for i in range(input_dim):
            embedder = nn.Sequential(
                nn.Linear(1, d_embed),
                nn.ReLU(),
                nn.LayerNorm(d_embed),
                nn.Dropout(dropout)
            )
            self.feature_embedders.append(embedder)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_embed))
        
        # Learnable feature type embeddings (geometric vs uncertainty features)
        self.type_embeddings = nn.Parameter(torch.randn(2, d_embed))  # 0: geometric, 1: uncertainty
        
        # Positional encodings for features
        self.pos_embeddings = nn.Parameter(torch.randn(1, input_dim + 1, d_embed))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_embed * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Attention pooling for final representation
        self.attention_pool = nn.Sequential(
            nn.Linear(d_embed, 1),
            nn.Softmax(dim=1)
        )
        
        # Output head with residual connections
        self.pre_output = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(d_embed * 2, d_embed),  # Concat CLS and pooled
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with careful initialization."""
        nn.init.normal_(self.cls_token, 0, 0.02)
        nn.init.normal_(self.pos_embeddings, 0, 0.02)
        nn.init.normal_(self.type_embeddings, 0, 0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def embed_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed each feature independently.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            embeddings: Feature embeddings [batch_size, input_dim, d_embed]
        """
        batch_size = x.size(0)
        feature_embeds = []
        
        for i, embedder in enumerate(self.feature_embedders):
            # Embed each feature
            feat_embed = embedder(x[:, i:i+1])  # [batch_size, d_embed]
            
            # Add feature type embedding (first 13 are geometric, last 4 are uncertainty)
            if i < 13:  # Geometric features
                feat_embed = feat_embed + self.type_embeddings[0]
            else:  # Uncertainty features
                feat_embed = feat_embed + self.type_embeddings[1]
            
            feature_embeds.append(feat_embed.unsqueeze(1))
        
        return torch.cat(feature_embeds, dim=1)  # [batch_size, input_dim, d_embed]
    
    def apply_mixup(self, embeddings: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
        """
        Apply mixup augmentation to embeddings.
        
        Args:
            embeddings: Feature embeddings [batch_size, seq_len, d_embed]
            alpha: Mixup interpolation strength
            
        Returns:
            mixed_embeddings: Mixed embeddings
        """
        if not self.training or not self.use_mixup:
            return embeddings
        
        batch_size = embeddings.size(0)
        
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample().to(embeddings.device)
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(embeddings.device)
        
        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        
        return mixed_embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SAINT-s.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            widths: Predicted interval widths [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Embed each feature
        feature_embeds = self.embed_features(x)  # [batch_size, input_dim, d_embed]
        
        # Apply mixup augmentation
        feature_embeds = self.apply_mixup(feature_embeds)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeds = torch.cat([cls_tokens, feature_embeds], dim=1)  # [batch_size, input_dim+1, d_embed]
        
        # Add positional embeddings
        embeds = embeds + self.pos_embeddings
        
        # Apply transformer
        transformer_out = self.transformer(embeds)  # [batch_size, input_dim+1, d_embed]
        
        # Extract CLS token output
        cls_output = transformer_out[:, 0]  # [batch_size, d_embed]
        
        # Attention pooling over feature tokens
        feature_tokens = transformer_out[:, 1:]  # [batch_size, input_dim, d_embed]
        attention_weights = self.attention_pool(feature_tokens)  # [batch_size, input_dim, 1]
        pooled_features = (feature_tokens * attention_weights).sum(dim=1)  # [batch_size, d_embed]
        
        # Process before final output
        cls_processed = self.pre_output(cls_output)
        pooled_processed = self.pre_output(pooled_features)
        
        # Combine CLS and pooled representations
        combined = torch.cat([cls_processed, pooled_processed], dim=1)  # [batch_size, d_embed*2]
        
        # Final prediction
        raw_output = self.output_head(combined)
        
        # Ensure positive output
        return self.ensure_positive_output(raw_output)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'd_embed': self.d_embed,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout': self.dropout_rate,
            'use_mixup': self.use_mixup,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "SAINT-s"