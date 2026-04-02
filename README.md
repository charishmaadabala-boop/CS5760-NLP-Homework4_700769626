# CS5760-NLP-Homework4_700769626
# Homework 4 Submission (Q1–Q3)
# Name: CHARISHMA ADABALA  
# Student ID: 700769626  

This assignment covers fundamental NLP deep learning concepts including:

- Character-level RNN Language Model
- Transformer Encoder architecture
- Scaled Dot-Product Attention
- Neural network theory questions (RNN, LSTM, Transformers)

All implementations are written in **PyTorch**.
# Q1
-----------------------------------------------------------
# Import required PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# DATA PREPARATION
# -----------------------------

# Small toy dataset for character-level language modeling
text = "hello hello help hello"

# Create vocabulary of unique characters
chars = list(set(text))

# Mapping from character to index (encoding)
char2idx = {c:i for i,c in enumerate(chars)}

# Mapping from index back to character (decoding)
idx2char = {i:c for c,i in char2idx.items()}

# Convert full text into numerical representation
data = [char2idx[c] for c in text]


# -----------------------------
# MODEL DEFINITION (RNN)
# -----------------------------

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        # Embedding layer: converts character indices into dense vectors
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Vanilla RNN layer for sequential processing
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # Fully connected layer maps hidden state to vocabulary space
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Convert input indices to embeddings
        x = self.embed(x)

        # Pass through RNN (captures sequential dependencies)
        out, h = self.rnn(x, h)

        # Map hidden states to character logits
        out = self.fc(out)

        return out, h


# -----------------------------
# TRAINING SETUP
# -----------------------------

model = CharRNN(len(chars), 128)

# Cross entropy loss for multi-class classification (next character prediction)
criterion = nn.CrossEntropyLoss()

# Adam optimizer for efficient gradient updates
optimizer = optim.Adam(model.parameters(), lr=0.01)


# -----------------------------
# TRAINING LOOP
# -----------------------------

for epoch in range(10):
    total_loss = 0
    h = None  # hidden state reset per epoch

    for i in range(len(data)-1):

        # Input character (t)
        x = torch.tensor([[data[i]]])

        # Target is next character (t+1)
        y = torch.tensor([data[i+1]])

        # Forward pass
        out, h = model(x, h)

        # Compute loss between prediction and true next character
        loss = criterion(out.view(-1, len(chars)), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss)
    # Q2
    -------------------------------------------------------
    import torch
    import torch.nn as nn

# -----------------------------
# MINI TRANSFORMER ENCODER
# -----------------------------

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2):
        super().__init__()

        # Embedding layer converts token IDs into dense vectors
        self.embed = nn.Embedding(vocab_size, d_model)

        # Multi-head self-attention layer
        # Captures relationships between all tokens in sequence
        self.attn = nn.MultiheadAttention(d_model, nhead)

        # Feed-forward network applied after attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),  # Non-linearity for feature transformation
            nn.Linear(128, d_model)
        )

        # Layer normalization improves training stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # Convert token IDs into embeddings
        x = self.embed(x)

        # Change shape to (sequence_length, batch_size, embedding_dim)
        # required by PyTorch MultiheadAttention
        x = x.permute(1, 0, 2)

        # Self-attention operation (query = key = value)
        attn_out, _ = self.attn(x, x, x)

        # Residual connection + normalization (stabilizes gradients)
        x = self.norm1(x + attn_out)

        # Feed-forward transformation
        ff_out = self.ff(x)

        # Second residual connection + normalization
        x = self.norm2(x + ff_out)

        return x


# -----------------------------
# MODEL TESTING
# -----------------------------

model = MiniTransformer(vocab_size=50)

# Random batch: (batch_size=5, sequence_length=10)
inp = torch.randint(0, 50, (5, 10))

# Forward pass through transformer encoder
out = model(inp)

# Output shape: (sequence_length, batch_size, d_model)
print(out.shape)

Q3.
-----------------------------------------------------------------------
import torch
import torch.nn.functional as F

# -----------------------------
# SCALED DOT-PRODUCT ATTENTION
# -----------------------------

def attention(Q, K, V):
    # Get dimensionality of key vectors
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores (similarity between Q and K)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale scores to prevent large values that destabilize softmax
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Step 3: Convert scores into probability distribution
    weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values based on attention weights
    output = torch.matmul(weights, V)

    return output, weights


# -----------------------------
# TESTING ATTENTION MECHANISM
# -----------------------------

# Random Query, Key, Value tensors
Q = torch.randn(2, 4)
K = torch.randn(2, 4)
V = torch.randn(2, 4)

# Compute attention output and weights
out, w = attention(Q, K, V)

# Display results
print("Attention Weights:", w)
print("Output:", out)
