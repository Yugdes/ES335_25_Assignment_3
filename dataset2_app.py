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
# Keep your local path; normalize so it works on Windows/Linux
BASE_DIR = os.path.normpath(
    r".\Dataset_2_LinuxKernel\models"
)

# mapping for the 8 models by (context, embedding, activation) -> filename
MODEL_MAP = {
    (3, 32, "relu"): "model_1_ctx3_emb32_relu",
    (3, 32, "tanh"): "model_2_ctx3_emb32_tanh",
    (3, 64, "relu"): "model_3_ctx3_emb64_relu",
    (3, 64, "tanh"): "model_4_ctx3_emb64_tanh",
    (5, 32, "relu"): "model_5_ctx5_emb32_relu",
    (5, 32, "tanh"): "model_6_ctx5_emb32_tanh",
    (5, 64, "relu"): "model_7_ctx5_emb64_relu",
    (5, 64, "tanh"): "model_8_ctx5_emb64_tanh",
}

# -------------------------------
# HELPER FUNCTIONS & MODEL
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
        # Handles (batch, context_size) and (batch, seq_len, context_size)
        if x.dim() == 3:
            b, seq_len, ctx = x.shape
            x = x.view(b * seq_len, ctx)
            embeds = self.embedding(x)
            flat = embeds.view(b * seq_len, -1)
            out = self.fc1(flat)
            out = self.act(out)
            logits = self.fc2(out)
            logits = logits.view(b, seq_len, self.vocab_size)
            return logits
        else:
            embeds = self.embedding(x)
            flat = embeds.view(embeds.size(0), -1)
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

    # Parse vocab
    if isinstance(vocab, dict):
        word2idx = vocab.get("word2idx") or vocab.get("word_to_idx") or vocab.get("stoi")
        idx2word = vocab.get("idx2word") or vocab.get("idx_to_word") or vocab.get("itos")
    elif isinstance(vocab, (list, tuple)) and len(vocab) == 2:
        word2idx, idx2word = vocab
    else:
        raise ValueError("Unrecognized vocab format in .pkl file")

    vocab_size = len(word2idx)

    # ✅ Always use config values — never from sidebar
    context_size = int(
    config.get("context_size") or config.get("context_length") or 3)
    embedding_dim = int(config["embedding_dim"])
    activation = config.get("activation", "relu").lower()

    # Load weights
    model_path = prefix + "_weights.pth"
    loaded = torch.load(model_path, map_location=torch.device("cpu"))

    # Build model exactly as trained
    model = NeuralLM(vocab_size, context_size, embedding_dim, hidden_size=1024, activation=activation)

    # Try to load weights safely
    if isinstance(loaded, dict):
        cleaned = OrderedDict()
        for k, v in loaded.items():
            cleaned[k.replace("module.", "")] = v
        try:
            model.load_state_dict(cleaned, strict=True)
        except RuntimeError as e:
            st.warning(f"⚠️ Weight mismatch detected: {e}")
            # Fallback: load loosely if minor mismatch
            model.load_state_dict(cleaned, strict=False)
    elif isinstance(loaded, nn.Module):
        model = loaded
    else:
        raise TypeError(f"Unrecognized model file format: {type(loaded)}")

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
        ctx = generated[-context_size:]
        context_ids = words_to_ids(ctx, word2idx)
        if len(context_ids) < context_size:
            pad = [word2idx.get("<PAD>", 0)] * (context_size - len(context_ids))
            context_ids = pad + context_ids

        context_tensor = torch.tensor([context_ids], dtype=torch.long)  # (1, context_size)

        with torch.no_grad():
            logits = model(context_tensor)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
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
st.write("Predict the next *k* words using trained neural language models on the Linux Kernel dataset.")

# Sidebar Controls
st.sidebar.header("Model Settings (choose architecture)")

# Context radio (only two choices), embedding and activation selectors
context_choice = st.sidebar.radio("Context size", [3, 5], index=0, format_func=lambda x: f"Context = {x}")
embedding_dim_choice = st.sidebar.selectbox("Embedding Dimension", [32, 64], index=0)
activation_choice = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"], index=0)

temperature = st.sidebar.slider("Temperature (randomness)", 0.5, 2.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)
k = st.sidebar.slider("Number of words to predict", 1, 50, 10)

st.write("---")

user_input = st.text_area("Enter your input text:", "The kernel module", height=100)

# Derive the model key from the three selections
model_tuple = (int(context_choice), int(embedding_dim_choice), activation_choice.lower())
if model_tuple not in MODEL_MAP:
    st.error(f"No model available for the combo: {model_tuple}")
else:
    model_key = MODEL_MAP[model_tuple]
    st.sidebar.write(f"Model file: **{model_key}**")

if st.button("Generate Next Words"):
    with st.spinner("Generating..."):
        # load model & config / using the model's config ensures correct context
        model, vocab, config = load_model_components(model_key)
        # use the model's actual context size from its config to avoid mismatch
        context_size = int(config.get("context_size", context_choice))
        result = generate_next_words(model, vocab, user_input, k, temperature, context_size, seed)
        st.success("✅ Generated Text:")
        st.write(result)

st.caption("Tip: choose context, embedding and activation to select a trained model. The app will auto-load the correct model file.")
