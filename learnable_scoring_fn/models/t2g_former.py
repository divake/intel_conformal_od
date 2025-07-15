"""T2G-Former (Tree-to-Graph Former) with graph-based feature interactions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseScoringFunction


class T2GFormerScoringFunction(BaseScoringFunction):
    """T2G-Former with learnable feature interaction graphs.
    
    T2G-Former learns a graph structure over features and uses this
    to guide attention patterns in transformer layers.
    """
    
    def __init__(self, input_dim: int = 17, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, graph_hidden: int = 32, dropout: float = 0.1,
                 graph_sparsity: float = 0.3,
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Number of input features
            d_model: Dimension of transformer model
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            graph_hidden: Hidden dimension for graph estimation
            dropout: Dropout probability
            graph_sparsity: Target sparsity for learned graph (0-1)
                    scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.graph_hidden = graph_hidden
        self.dropout_rate = dropout
        self.graph_sparsity = graph_sparsity
        
        # Graph Estimator Network
        # Static feature embeddings for graph construction
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, graph_hidden))
        
        # Dynamic graph estimation based on input
        self.graph_mlp = nn.Sequential(
            nn.Linear(input_dim, graph_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, graph_hidden),
            nn.ReLU(),
            nn.Linear(graph_hidden, input_dim * input_dim)
        )
        
        # Feature projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encodings for features
        self.pos_encodings = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Graph-guided Transformer blocks
        self.transformer_layers = nn.ModuleList([
            GraphGuidedTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Cross-level feature aggregation
        self.feature_aggregator = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Final readout network
        self.readout = nn.Sequential(
            nn.Linear(d_model * (n_layers + 1), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        nn.init.xavier_uniform_(self.feature_embeddings)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def estimate_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate feature relation graph.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            graphs: Adjacency matrices [batch_size, input_dim, input_dim]
        """
        batch_size = x.size(0)
        
        # Static graph from feature embeddings
        static_graph = torch.matmul(self.feature_embeddings, self.feature_embeddings.T)
        static_graph = F.softmax(static_graph / self.graph_hidden**0.5, dim=-1)
        
        # Dynamic graph weights based on input
        dynamic_weights = self.graph_mlp(x).view(batch_size, self.input_dim, self.input_dim)
        
        # Combine static structure with dynamic weights
        graphs = torch.sigmoid(static_graph.unsqueeze(0) + 0.1 * dynamic_weights)
        
        # Apply sparsity constraint (soft thresholding)
        if self.training:
            # During training, use soft sparsity
            graphs = graphs * (graphs > self.graph_sparsity).float()
        else:
            # During inference, hard threshold
            graphs = (graphs > self.graph_sparsity).float()
        
        # Ensure graph is symmetric
        graphs = 0.5 * (graphs + graphs.transpose(-1, -2))
        
        # Add self-loops
        eye = torch.eye(self.input_dim, device=graphs.device).unsqueeze(0)
        graphs = graphs + eye
        
        # Normalize (row-wise)
        graphs = graphs / (graphs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return graphs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through T2G-Former.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            widths: Predicted interval widths [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Estimate feature relation graph
        fr_graphs = self.estimate_graph(x)  # [batch_size, input_dim, input_dim]
        
        # Project features 
        h = self.input_projection(x)  # [batch_size, d_model]
        # Reshape to [batch_size, input_dim, d_model/input_dim] if needed
        # But since input_projection outputs [batch_size, d_model], we need to handle this differently
        
        # Create feature tokens by reshaping
        h = h.unsqueeze(1)  # [batch_size, 1, d_model]
        # Repeat for each input dimension
        h = h.expand(-1, self.input_dim, -1)  # [batch_size, input_dim, d_model]
        
        # Add feature-specific transformations
        for i in range(self.input_dim):
            h[:, i, :] = h[:, i, :] + x[:, i:i+1] * 0.1  # Add residual from input
        
        # Add positional encodings
        h = h + self.pos_encodings
        
        # Collect representations from all layers for cross-level aggregation
        layer_outputs = [h]
        
        # Process through graph-guided transformer layers
        for layer in self.transformer_layers:
            h = layer(h, fr_graphs)
            layer_outputs.append(h)
        
        # Cross-level feature aggregation
        # Simply concatenate representations from each layer
        layer_representations = []
        for layer_output in layer_outputs:
            # Global average pooling for each layer
            pooled = layer_output.mean(dim=1)  # [batch_size, d_model]
            layer_representations.append(pooled)
        
        # Concatenate all layer representations
        combined = torch.cat(layer_representations, dim=1)  # [batch_size, d_model * (n_layers + 1)]
        
        # Final readout
        raw_output = self.readout(combined)
        
        # Ensure positive output
        return self.ensure_positive_output(raw_output)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'graph_hidden': self.graph_hidden,
            'dropout': self.dropout_rate,
            'graph_sparsity': self.graph_sparsity
        ,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "T2G-Former"


class GraphGuidedTransformerLayer(nn.Module):
    """Single graph-guided transformer layer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Graph-guided gating
        self.graph_gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.Sigmoid()
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph-guided transformer layer.
        
        Args:
            x: Features [batch_size, input_dim, d_model]
            graph: Adjacency matrix [batch_size, input_dim, input_dim]
            
        Returns:
            output: Transformed features [batch_size, input_dim, d_model]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        
        # Graph-guided modulation
        # Aggregate neighbor information using graph
        graph_aggregated = torch.bmm(graph, x)  # [batch_size, input_dim, d_model]
        
        # Gate based on graph information
        gate_input = torch.cat([attn_out, graph_aggregated], dim=-1)
        gate = self.graph_gate(gate_input)
        
        # Apply gating
        attn_out = gate * attn_out + (1 - gate) * graph_aggregated
        
        # Residual and norm
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x