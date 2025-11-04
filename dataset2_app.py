# app_linuxkernel.py
import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import json
import os
import random
import numpy as np
from collections import OrderedDict
import torch.nn as nn
# -------------------------------
# CONFIGURE PATHS
# -------------------------------
BASE_DIR = r"D:\Onedrive\OneDrive - iitgn.ac.in\Desktop\Task_1_ML_assignment_3\Dataset_2_LinuxKernel\models"

# Model name mapping
MODEL_OPTIONS = {
    "Context=3 | Embedding=32 | ReLU": "model_1_ctx3_emb32_relu",
    "Context=3 | Embedding=32 | Tanh": "model_2_ctx3_emb32_tanh",
    "Context=3 | Embedding=64 | ReLU": "model_3_ctx3_emb64_relu",
    "Context=3 | Embedding=64 | Tanh": "model_4_ctx3_emb64_tanh",
    "Context=5 | Embedding=32 | ReLU": "model_5_ctx5_emb32_relu",
    "Context=5 | Embedding=32 | Tanh": "model_6_ctx5_emb32_tanh",
    "Context=5 | Embedding=64 | ReLU": "model_7_ctx5_emb64_relu",
    "Context=5 | Embedding=64 | Tanh": "model_8_ctx5_emb64_tanh",
}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, hidden_size=1024, activation='relu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.act = nn.ReLU() if activation.lower() == 'relu' else nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """
        Forward pass for both (batch, context_size) and (batch, seq_len, context_size)
        """
        if x.dim() == 3:
            # (batch, seq_len, context_size)
            b, seq_len, ctx = x.shape
            x = x.view(b * seq_len, ctx)

            embeds = self.embedding(x)               # (b*seq_len, ctx, emb)
            flat = embeds.view(b * seq_len, -1)      # flatten
            out = self.fc1(flat)
            out = self.act(out)
            logits = self.fc2(out)
            logits = logits.view(b, seq_len, self.vocab_size)
            return logits
        else:
            # (batch, context_size)
            embeds = self.embedding(x)               # (batch, context, emb)
            flat = embeds.view(embeds.size(0), -1)   # flatten
            out = self.fc1(flat)
            out = self.act(out)
            logits = self.fc2(out)
            return logits

def load_model_components(model_key):
    prefix = os.path.join(BASE_DIR, model_key)

    # Load vocab + config
    with open(prefix + "_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(prefix + "_config.json", "r") as f:
        config = json.load(f)

    # --- FIX: handle alternate key names ---
    if isinstance(vocab, dict):
        word2idx = vocab.get("word2idx") or vocab.get("word_to_idx") or vocab.get("stoi")
        idx2word = vocab.get("idx2word") or vocab.get("idx_to_word") or vocab.get("itos")
        if word2idx is None or idx2word is None:
            raise KeyError("Could not find word2idx / idx2word in vocab dictionary.")
    elif isinstance(vocab, (list, tuple)) and len(vocab) == 2:
        word2idx, idx2word = vocab
        vocab = {"word2idx": word2idx, "idx2word": idx2word}
    else:
        raise ValueError("Unrecognized vocab format in .pkl file")

    vocab_size = len(word2idx)
    context_size = int(config.get("context_size", 3))
    embedding_dim = int(config.get("embedding_dim", 32))
    activation = config.get("activation", "relu")

    # Load model weights
    model_path = prefix + "_weights.pth"
    loaded = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        model = NeuralLM(vocab_size, context_size, embedding_dim, hidden_size=1024, activation=activation)
        model.load_state_dict(loaded)

    model.eval()
    vocab = {"word2idx": word2idx, "idx2word": idx2word}
    return model, vocab, config


def words_to_ids(words, word2idx, unk_token="<UNK>"):
    return [word2idx.get(w, word2idx.get(unk_token, 0)) for w in words]


def ids_to_words(ids, idx2word):
    return [idx2word.get(i, "<UNK>") for i in ids]


def generate_next_words(model, vocab, prompt, k=10, temperature=1.0, context_size=3, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    words = prompt.strip().split()
    generated = words.copy()

    for _ in range(k):
        # prepare context ids (pad with UNK or zeros if needed)
        ctx = generated[-context_size:]
        context_ids = words_to_ids(ctx, word2idx)
        # pad if shorter than context_size
        if len(context_ids) < context_size:
            pad = [word2idx.get("<PAD>", 0)] * (context_size - len(context_ids))
            context_ids = pad + context_ids

        context_tensor = torch.tensor([context_ids], dtype=torch.long)  # (1, context_size)

        with torch.no_grad():
            logits = model(context_tensor)  # either (1, vocab) or (1, seq_len, vocab)

            if logits.dim() == 3:
                # use last time-step
                logits = logits[:, -1, :]
            else:
                logits = logits  # already (1, vocab)

            logits = logits / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        next_word = idx2word.get(next_id, "<UNK>")
        generated.append(next_word)

    return " ".join(generated)


# -------------------------------
# STREAMLIT APP
# -------------------------------

st.set_page_config(page_title="Linux Kernel Text Generator", layout="centered")

st.title("🐧 Linux Kernel Text Generator")
st.write("This app predicts the next *k* words using trained neural language models on Linux Kernel dataset.")

# Sidebar Controls
st.sidebar.header("Model Configuration")

model_choice = st.sidebar.selectbox("Select Model Variant", list(MODEL_OPTIONS.keys()))
context_size = st.sidebar.slider("Context Length", 1, 10, 3)
embedding_dim = st.sidebar.selectbox("Embedding Dimension", [32, 64])
activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"])
temperature = st.sidebar.slider("Temperature (randomness)", 0.5, 2.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)
k = st.sidebar.slider("Number of words to predict", 1, 50, 10)

st.write("---")

# Text input from user
user_input = st.text_area("Enter your input text:", "The kernel module", height=100)

if st.button("Generate Next Words"):
    with st.spinner("Generating..."):
        model_key = MODEL_OPTIONS[model_choice]
        model, vocab, config = load_model_components(model_key)
        result = generate_next_words(model, vocab, user_input, k, temperature, context_size, seed)
        st.success("✅ Generated Text:")
        st.write(result)

st.caption("Tip: Lower temperature makes output more deterministic; higher values make it more random.")
