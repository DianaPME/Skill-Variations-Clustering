import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

# ============ GCN Module ============

class SimpleVocabGCN(nn.Module):
    """Simplified GCN for vocabulary enhancement of embeddings"""
    def __init__(self, vocab_size, embed_dim, hidden_dim=128):
        super(SimpleVocabGCN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Simple linear layers for GCN
        self.W1 = nn.Linear(vocab_size, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, tfidf_features, adj_matrix):
        """
        Args:
            tfidf_features: (batch_size, vocab_size) - TF-IDF features
            adj_matrix: (vocab_size, vocab_size) - vocabulary adjacency matrix
        """
        # Handle sparse matrices
        if hasattr(tfidf_features, 'toarray'):  
            tfidf_features = tfidf_features.toarray()
        if hasattr(adj_matrix, 'toarray'):  
            adj_matrix = adj_matrix.toarray()

        # Ensure inputs are tensors
        if isinstance(tfidf_features, np.ndarray):
            tfidf_features = torch.tensor(tfidf_features, dtype=torch.float32)
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        # Simple GCN operation: X * A * W
        gcn_features = torch.matmul(tfidf_features, adj_matrix) 

        # Apply two linear transformations
        hidden = self.relu(self.W1(gcn_features))  
        hidden = self.dropout(hidden)
        output = self.W2(hidden)  

        return output