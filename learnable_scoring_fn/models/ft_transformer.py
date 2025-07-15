"""FT-Transformer (Feature Tokenizer + Transformer) for interval width prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseScoringFunction


class FTTransformerScoringFunction(BaseScoringFunction):
    """Lightweight FT-Transformer for interval width prediction.
    
    FT-Transformer treats each feature as a token and uses self-attention
    to model feature interactions. This is particularly effective for
    tabular data with mixed feature types.
    """
    
    def __init__(self, input_dim: int = 17, n_blocks: int = 2, d_block: int = 64, 
                 n_heads: int = 4, ffn_factor: float = 1.0, dropout: float = 0.1,
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Number of input features
            n_blocks: Number of transformer blocks
            d_block: Dimension of transformer blocks
            n_heads: Number of attention heads
            ffn_factor: FFN dimension multiplier (ffn_dim = d_block * ffn_factor)
            dropout: Dropout probability
            scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        # Save config
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.n_heads = n_heads
        self.ffn_factor = ffn_factor
        self.dropout_rate = dropout
        
        # Feature tokenizer - embed each feature independently
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_block),
                nn.LayerNorm(d_block)
            ) for _ in range(input_dim)
        ])
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_block))
        
        # Positional embeddings for features (optional but can help)
        self.pos_embeddings = nn.Parameter(torch.randn(1, input_dim + 1, d_block))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_block,
            nhead=n_heads,
            dim_feedforward=int(d_block * ffn_factor),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture tends to be more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_blocks)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_block),
            nn.Linear(d_block, d_block // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_block // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FT-Transformer.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            widths: Predicted interval widths [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Tokenize each feature independently
        feature_tokens = []
        for i, embed in enumerate(self.feature_embeddings):
            # Each feature is embedded separately
            token = embed(x[:, i:i+1])  # [batch_size, d_block]
            feature_tokens.append(token.unsqueeze(1))  # [batch_size, 1, d_block]
        
        feature_tokens = torch.cat(feature_tokens, dim=1)  # [batch_size, input_dim, d_block]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_block]
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)  # [batch_size, input_dim+1, d_block]
        
        # Add positional embeddings
        tokens = tokens + self.pos_embeddings
        
        # Apply transformer
        output = self.transformer(tokens)  # [batch_size, input_dim+1, d_block]
        
        # Use CLS token for prediction
        cls_output = output[:, 0]  # [batch_size, d_block]
        
        # Predict width
        raw_output = self.output_head(cls_output)  # [batch_size, 1]
        
        # Ensure positive output
        return self.ensure_positive_output(raw_output)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'n_blocks': self.n_blocks,
            'd_block': self.d_block,
            'n_heads': self.n_heads,
            'ffn_factor': self.ffn_factor,
            'dropout': self.dropout_rate,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "FT-Transformer"