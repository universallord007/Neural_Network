# GPT-2 From Scratch with Mixture of Experts (MoE)

A from-scratch implementation of a GPT-2-style Transformer in PyTorch, extended with a **Mixture of Experts (MoE)** architecture.

The goal of this project is to understand how modern language models work internally by implementing the major components rather than relying entirely on high-level abstractions.

## 🚀 Features

- GPT-2-style Transformer architecture
- Causal Self-Attention
- Multi-Head Attention
- Scaled Dot-Product Attention
- Token and positional embeddings
- Layer Normalization
- GELU activation
- Weight tying between token embeddings and language-model head
- Mixture of Experts (MoE)
- Top-K routing
- 4 independent expert MLPs
- Router/gating network
- MoE load-balancing auxiliary loss
- Top-50 token sampling during generation
- GPT-2 BPE tokenizer using `tiktoken`
- Mixed-precision training with `bfloat16`
- Gradient clipping
- AdamW optimization
- Learning-rate warmup/decay
- PyTorch `torch.compile()`

---

## 🧠 Architecture

```text
Input Tokens
     │
     ▼
Token Embeddings + Positional Embeddings
     │
     ▼
┌─────────────────────────┐
│      Transformer Block  │
│                         │
│ LayerNorm               │
│      ↓                  │
│ Causal Self-Attention   │
│      ↓                  │
│ Residual Connection     │
│      ↓                  │
│ LayerNorm               │
│      ↓                  │
│ Mixture of Experts      │
│      ↓                  │
│ Residual Connection     │
└─────────────────────────┘
     │
     ▼
   LayerNorm
     │
     ▼
 Language Model Head
     │
     ▼
   Logits

The Transformer block uses an MoE layer instead of a standard single MLP.

🔀 Mixture of Experts

The main experiment in this project is replacing the normal Transformer MLP with multiple expert networks.

Each expert is a small feed-forward network:

Input
  ↓
Linear
  ↓
GELU
  ↓
Linear
  ↓
Output

The router determines which experts should process each token.

Token Representation
        │
        ▼
      Router
        │
        ▼
 Expert Probabilities
        │
        ▼
      Top-K
        │
   ┌────┴────┐
   ▼         ▼
Expert 1   Expert 2
   │         │
   └────┬────┘
        ▼
 Weighted Combination
        │
        ▼
     MoE Output
Current Configuration
Number of experts: 4
Top-K experts per token: 2

The router produces one probability for each expert and only the top 2 experts are selected for each token.

⚖️ Load Balancing Loss

A router can potentially send most tokens to only a few experts.

To encourage more balanced expert utilization, the project implements an auxiliary load-balancing loss:

L_balance = N × Σ(f_i × P_i)

Where:

N = number of experts
f_i = fraction of tokens routed to expert i
P_i = average routing probability for expert i

The final training objective is:

Total Loss = Language Model Loss + λ × L_balance

The current implementation uses:

λ = 0.1
🏗️ Main Components
CausalSelfAttention

Implements causal multi-head self-attention using PyTorch's:

F.scaled_dot_product_attention()

This prevents tokens from attending to future tokens.

Expert

Each expert is an independent MLP:

Linear → GELU → Linear
MoE

Responsible for:

Router predictions
Softmax probabilities
Top-K expert selection
Routing weights
Expert execution
Weighted expert outputs
Load-balancing loss
Block

Each Transformer block contains:

LayerNorm
→ Causal Self-Attention
→ Residual Connection
→ LayerNorm
→ MoE
→ Residual Connection
GPT

The main language model containing:

Token embeddings
Positional embeddings
Transformer blocks
Final LayerNorm
Language-model head
✍️ Text Generation

The model uses GPT-2's tokenizer through tiktoken.

Example prompt:

Hello, I am a language model

Generation uses Top-50 sampling:

Logits
  ↓
Softmax
  ↓
Top 50 tokens
  ↓
Randomly sample one token
  ↓
Append token
  ↓
Repeat

This allows the model to generate multiple possible continuations instead of always selecting the single highest-probability token.

🛠️ Tech Stack
Python
PyTorch
Tiktoken
CUDA
GPT-2 BPE Tokenizer
📦 Installation

Clone the repository:

git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

Install dependencies:

pip install torch tiktoken

If using CUDA, make sure PyTorch is installed with the appropriate CUDA build for your system.

▶️ Running the Project

Place your training text inside:

input.txt

Then run the Python script/notebook.

The training pipeline:

input.txt
   ↓
GPT-2 Tokenizer
   ↓
Token Batches
   ↓
GPT + MoE
   ↓
Cross Entropy Loss
   +
Load Balancing Loss
   ↓
Backpropagation
   ↓
AdamW
📊 Training Output

During training, the model reports:

Training loss
Gradient norm
Step execution time
Tokens processed per second

Example:

Loss | 6.8666
time diff | 560.66ms
norm | 3.4540
🎯 Project Goals

This project is primarily an educational implementation designed to understand:

How GPT-style Transformers work internally
How self-attention is implemented
How tokens flow through a Transformer
How Mixture of Experts routing works
How Top-K routing affects computation
How expert utilization can be measured
Why load balancing is needed in MoE models
How language-model training and generation work
🔬 Future Improvements
 Better expert utilization analysis
 Visualization of router decisions
 Track expert selection frequencies during training
 Experiment with different top_k values
 Experiment with different numbers of experts
 Temperature-based sampling
 Top-P (nucleus) sampling
 Expert capacity limits
 More efficient batched expert computation
 Compare standard MLP vs MoE
 Train for longer and evaluate generated text
 Write a technical breakdown of the MoE routing mechanism
📚 Inspiration

This project was built as a learning exercise while studying GPT architectures and implementing Transformer components from scratch.

Special inspiration from:

Andrej Karpathy's Zero to Hero series
GPT-2 architecture
Modern Mixture-of-Experts language models
👨‍💻 Author

Dipankar Dutta

Building and learning AI/ML systems from first principles.

⭐ If you find this project useful

Feel free to explore the implementation, experiment with the routing mechanism, and modify the architecture.