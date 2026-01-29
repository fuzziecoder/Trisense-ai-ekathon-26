"""
TriSense AI - PatchTST Encoder for Time-Series Embeddings
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import os
import json

from ..config import settings


class PatchTSTEncoderModel(nn.Module):
    """
    PatchTST Encoder architecture for time-series vital signs.
    Generates 32-dimensional embeddings from a window of vital readings.
    """
    
    def __init__(
        self,
        c_in: int = 6,          # Number of vital sign channels
        patch_len: int = 2,      # Patch length
        stride: int = 1,         # Stride
        d_model: int = 64,       # Model dimension
        n_heads: int = 4,        # Attention heads
        n_layers: int = 2,       # Transformer layers
        d_ff: int = 128,         # Feed-forward dimension
        dropout: float = 0.1,
        embed_dim: int = 32      # Output embedding dimension
    ):
        super().__init__()
        
        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * c_in, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embed_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, c_in)
               seq_len = window_size (e.g., 6)
               c_in = number of vital signs (6)
        
        Returns:
            Embeddings of shape (batch, embed_dim)
        """
        batch_size, seq_len, c_in = x.shape
        
        # Create patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :].reshape(batch_size, -1)
            patches.append(patch)
        
        # Stack patches
        if patches:
            x = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * c_in)
        else:
            # If seq_len is too short, flatten and project directly
            x = x.reshape(batch_size, 1, -1)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :x.size(1), :]
        x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, num_patches, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        x = self.output_proj(x)  # (batch, embed_dim)
        x = self.layer_norm(x)
        
        return x


class PatchTSTEncoder:
    """
    Wrapper class for PatchTST encoder with model loading and inference.
    """
    
    def __init__(self):
        self.model: Optional[PatchTSTEncoderModel] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_dim = 32
        self.window_size = 6
        self.vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp',
            'respiratory_rate', 'spo2', 'temperature'
        ]
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained PatchTST encoder"""
        model_path = os.path.join(settings.MODEL_DIR, settings.PATCHTST_MODEL)
        
        try:
            if os.path.exists(model_path):
                self.model = PatchTSTEncoderModel()
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ PatchTST encoder loaded from {model_path}")
            else:
                print(f"⚠️ Model not found at {model_path}, using random initialization")
                self.model = PatchTSTEncoderModel()
                self.model.to(self.device)
                self.model.eval()
        except Exception as e:
            print(f"⚠️ Error loading PatchTST model: {e}, using random initialization")
            self.model = PatchTSTEncoderModel()
            self.model.to(self.device)
            self.model.eval()
    
    def encode(self, vitals_window: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate embeddings from a window of vital signs.
        
        Args:
            vitals_window: Dict mapping vital name to array of values
                           Each array should have `window_size` elements
        
        Returns:
            Embedding array of shape (embed_dim,)
        """
        if self.model is None:
            return np.zeros(self.embed_dim)
        
        # Prepare input tensor
        try:
            # Stack vitals into (window_size, num_vitals) array
            vital_arrays = []
            for col in self.vital_columns:
                arr = vitals_window.get(col, np.zeros(self.window_size))
                if len(arr) < self.window_size:
                    # Pad with mean or zeros
                    padded = np.zeros(self.window_size)
                    padded[-len(arr):] = arr
                    arr = padded
                elif len(arr) > self.window_size:
                    arr = arr[-self.window_size:]
                vital_arrays.append(arr)
            
            # Shape: (window_size, num_vitals)
            input_data = np.stack(vital_arrays, axis=1)
            
            # Normalize
            input_data = self._normalize(input_data)
            
            # Convert to tensor
            x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # (1, window_size, num_vitals)
            x = x.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(x)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error encoding vitals: {e}")
            return np.zeros(self.embed_dim)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize vital signs to reasonable ranges"""
        # Normalization ranges for each vital
        ranges = {
            0: (40, 180),    # heart_rate
            1: (60, 200),    # systolic_bp
            2: (40, 120),    # diastolic_bp
            3: (8, 40),      # respiratory_rate
            4: (70, 100),    # spo2
            5: (35, 42)      # temperature
        }
        
        normalized = data.copy()
        for i, (min_val, max_val) in ranges.items():
            if i < normalized.shape[1]:
                normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        
        return normalized


# Singleton instance
_encoder_instance: Optional[PatchTSTEncoder] = None


def get_encoder() -> PatchTSTEncoder:
    """Get or create encoder instance"""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = PatchTSTEncoder()
    return _encoder_instance
