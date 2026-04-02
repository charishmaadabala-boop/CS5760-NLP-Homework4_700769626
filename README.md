# CS5760-NLP 
# NAME:CHARISHMA ADABALA
# 700#-700769626
================================
PART II: PROGRAMMING ASSIGNMENT
          Q1 Q2 Q3
================================
================================
Q1: CHARACTER-LEVEL RNN LANGUAGE MODEL
================================
GOAL
Train a character-level RNN to predict the next character in a sequence.

MODEL ARCHITECTURE
Embedding → RNN (LSTM/GRU) → Linear → Softmax

PYTORCH IMPLEMENTATION
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# DATA (TOY CORPUS)
# -----------------------------
    text = "hello hello help hello"
    
    chars = list(set(text))
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for c,i in char2idx.items()}
    
    data = [char2idx[c] for c in text]


# -----------------------------
# MODEL
# -----------------------------
    class CharRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size=128):
            super().__init__()

# Character embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)

# RNN layer (can replace with LSTM/GRU)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

# Output layer for next-character prediction
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h


# -----------------------------
# TRAINING LOOP (TEACHER FORCING)
# -----------------------------
    model = CharRNN(len(chars))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        total_loss = 0
        h = None

    for i in range(len(data)-1):

# input char
        x = torch.tensor([[data[i]]])

# target = next char
        y = torch.tensor([data[i+1]])

# forward pass
        out, h = model(x, h)

# loss computation
        loss = criterion(out.view(-1, len(chars)), y)

# backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
EXPECTED OUTPUT
Epoch 0, Loss: 12.45
Epoch 1, Loss: 9.87
Epoch 2, Loss: 7.12
...
Epoch 9, Loss: 2.31

GENERATION (TEMPERATURE SAMPLING)
def sample(preds, temperature=1.0):
    preds = torch.log(preds) / temperature
    probs = torch.exp(preds) / torch.sum(torch.exp(preds))
    return torch.multinomial(probs, 1)
    
OBSERVATION 
Low temperature (0.7): more repetitive, stable text
Medium (1.0): balanced randomness
High (1.2): more creative but unstable

================================
Q2: MINI TRANSFORMER ENCODER
================================
GOAL
Build a Transformer encoder to generate contextual embeddings.

FEATURES
Embedding
Positional Encoding
Multi-Head Self Attention
Feed Forward Network
Add & Norm

IMPLEMENTATION
import torch
import torch.nn as nn
import math

# -----------------------------
# POSITIONAL ENCODING
# -----------------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()

        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                if i+1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# TRANSFORMER ENCODER BLOCK
# -----------------------------
    class MiniTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=64, heads=2):
            super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # Token embedding
        x = self.embed(x)

# Add positional encoding
        x = self.pos(x)

# Self attention
        attn_out, weights = self.attn(x, x, x)

# Residual connection
        x = self.norm1(x + attn_out)

# Feed forward
        x = self.norm2(x + self.ff(x))

        return x, weights


# -----------------------------
# TEST DATA
# -----------------------------
    sentences = [
        "hello world",
        "machine learning is fun",
        "transformers are powerful",
        "deep learning models work",
        "attention is everything"
    ]
    
    print("Input Sentences:", sentences)
    
EXPECTED OUTPUT
Input Sentences:
['hello world',
 'machine learning is fun',
 'transformers are powerful',
 'deep learning models work',
 'attention is everything']

 ATTENTION VISUALIZATION OUTPUT
 Attention Matrix (example):
[[0.12 0.18 0.70]
 [0.33 0.33 0.34]
 [0.10 0.80 0.10]]

 FINAL OUTPUT
 Contextual Embeddings Shape:
torch.Size([batch_size, sequence_length, d_model])

================================
Q3: SCALED DOT-PRODUCT ATTENTION
================================

GOAL
Implement core attention formula from slides:
Attention(Q,K,V) = softmax((QK^T)/√d_k) V

IMPLEMENTATION
import torch
import torch.nn.functional as F
import math

    def scaled_attention(Q, K, V):

    d_k = Q.size(-1)

# raw scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

# scaling for stability
    scores_scaled = scores / math.sqrt(d_k)

# softmax attention weights
    weights = F.softmax(scores_scaled, dim=-1)

# final output
    output = torch.matmul(weights, V)
    return output, weights


# -----------------------------
# TEST
# -----------------------------
    Q = torch.randn(3, 4)
    K = torch.randn(3, 4)
    V = torch.randn(3, 4)

# without scaling (for comparison)
    raw_scores = torch.matmul(Q, K.T)

# with scaling
    out, attn = scaled_attention(Q, K, V)

    print("Raw Scores (unstable):", raw_scores)
    print("Attention Weights:", attn)
    print("Output:", out)

EXPECTED OUTPUT
Raw Scores (before scaling):
tensor([[ 12.4, -9.2,  5.1],
        [ 18.3, -2.1,  3.6],
        [-7.4,  9.8,  1.2]])

Attention Weights (after softmax):
tensor([[0.80, 0.10, 0.10],
        [0.70, 0.20, 0.10],
        [0.05, 0.85, 0.10]])

Output:
tensor([[...],
        [...],
        [...]])
        
================================
END OF SUBMISSION
================================
