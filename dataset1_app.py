import streamlit as st
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Sherlock Holmes Text Generator",
    page_icon="🔍",
    layout="wide"
)

# Define base directory
BASE_DIR = Path(r"./Dataset_1_SherlockHolmes")

# Define MLP Model Architecture (3 hidden layers)
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, hidden_dim, activation='relu'):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.context_length = context_length
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Model configurations
MODELS = {
    "Context=3, Embedding=32, ReLU": {
        "folder": "mlp_ctx3_emb32_relu",
        "context": 3,
        "embedding": 32,
        "activation": "relu"
    },
    "Context=3, Embedding=32, Tanh": {
        "folder": "mlp_ctx3_emb32_tanh",
        "context": 3,
        "embedding": 32,
        "activation": "tanh"
    },
    "Context=3, Embedding=64, ReLU": {
        "folder": "mlp_ctx3_emb64_relu",
        "context": 3,
        "embedding": 64,
        "activation": "relu"
    },
    "Context=3, Embedding=64, Tanh": {
        "folder": "mlp_ctx3_emb64_tanh",
        "context": 3,
        "embedding": 64,
        "activation": "tanh"
    },
    "Context=5, Embedding=32, ReLU": {
        "folder": "mlp_ctx5_emb32_relu",
        "context": 5,
        "embedding": 32,
        "activation": "relu"
    },
    "Context=5, Embedding=32, Tanh": {
        "folder": "mlp_ctx5_emb32_tanh",
        "context": 5,
        "embedding": 32,
        "activation": "tanh"
    },
    "Context=5, Embedding=64, ReLU": {
        "folder": "mlp_ctx5_emb64_relu",
        "context": 5,
        "embedding": 64,
        "activation": "relu"
    },
    "Context=5, Embedding=64, Tanh": {
        "folder": "mlp_ctx5_emb64_tanh",
        "context": 5,
        "embedding": 64,
        "activation": "tanh"
    }
}

@st.cache_resource
def load_model_and_vocab(model_name):
    """Load model and vocabulary with caching"""
    config = MODELS[model_name]
    model_folder = BASE_DIR / config["folder"]
    
    # Load vocabulary
    vocab_files = list(model_folder.glob("*vocabulary.pkl"))
    if not vocab_files:
        vocab_files = list(model_folder.glob("vocabulary.pkl"))
    
    if not vocab_files:
        st.error(f"Vocabulary file not found in {model_folder}")
        return None, None, None
    
    with open(vocab_files[0], 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Extract word to index and index to word mappings
    if isinstance(vocab_data, dict):
        word_to_idx = vocab_data.get('word_to_idx', vocab_data)
        idx_to_word = vocab_data.get('idx_to_word', {v: k for k, v in word_to_idx.items()})
    else:
        word_to_idx = vocab_data
        idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    vocab_size = len(word_to_idx)
    
    # Load model
    model_files = list(model_folder.glob("sherlock_*.pt"))
    if not model_files:
        st.error(f"Model file not found in {model_folder}")
        return None, None, None
    
    # Initialize model architecture
    # All models use hidden_dim=1024 (constant across all 8 models)
    hidden_dim = 1024
    model = MLP(
        vocab_size=vocab_size,
        embedding_dim=config["embedding"],
        context_length=config["context"],
        hidden_dim=hidden_dim,
        activation=config["activation"]
    )
    
    # Load model weights
    checkpoint = torch.load(model_files[0], map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, word_to_idx, idx_to_word

def handle_oov_word(word, word_to_idx):
    """Handle out-of-vocabulary words"""
    # Try lowercase
    if word.lower() in word_to_idx:
        return word.lower()
    
    # Try capitalized
    if word.capitalize() in word_to_idx:
        return word.capitalize()
    
    # Try uppercase
    if word.upper() in word_to_idx:
        return word.upper()
    
    # Use unknown token or random word
    if '<unk>' in word_to_idx:
        return '<unk>'
    elif '<UNK>' in word_to_idx:
        return '<UNK>'
    else:
        # Return a random common word from vocabulary
        common_words = ['the', 'a', 'and', 'of', 'to', 'in', 'that', 'was']
        for cw in common_words:
            if cw in word_to_idx:
                return cw
        # Last resort: return first word in vocabulary
        return list(word_to_idx.keys())[0]

def generate_text(model, word_to_idx, idx_to_word, seed_text, num_words, temperature, context_length):
    """Generate text using the model with temperature control"""
    model.eval()
    
    # Tokenize and clean seed text
    words = seed_text.lower().split()
    
    # Handle case where seed text is shorter than context
    if len(words) < context_length:
        # Pad with common words
        padding_words = ['the'] * (context_length - len(words))
        words = padding_words + words
    
    # Handle OOV words in seed text
    processed_words = []
    oov_words = []
    for word in words[-context_length:]:
        if word not in word_to_idx:
            oov_words.append(word)
            processed_word = handle_oov_word(word, word_to_idx)
            processed_words.append(processed_word)
        else:
            processed_words.append(word)
    
    generated_words = processed_words.copy()
    
    with torch.no_grad():
        for _ in range(num_words):
            # Get context
            context = generated_words[-context_length:]
            
            # Convert to indices
            context_indices = [word_to_idx[w] for w in context]
            context_tensor = torch.tensor([context_indices])
            
            # Get model prediction
            output = model(context_tensor)
            
            # Apply temperature
            logits = output[0] / temperature
            probs = F.softmax(logits, dim=0).numpy()
            
            # Sample next word
            next_idx = np.random.choice(len(probs), p=probs)
            next_word = idx_to_word[next_idx]
            
            generated_words.append(next_word)
    
    return ' '.join(generated_words), oov_words

def load_training_summary(model_folder):
    """Load training summary"""
    summary_files = list(Path(BASE_DIR / model_folder).glob("*training_summary.txt"))
    if not summary_files:
        summary_files = list(Path(BASE_DIR / model_folder).glob("training_summary.txt"))
    
    if summary_files:
        with open(summary_files[0], 'r') as f:
            return f.read()
    return "Training summary not found"

def load_image(model_folder, image_type):
    """Load images"""
    if image_type == "training":
        pattern = "training_curve_*.png"
    else:
        pattern = "embeddings_visualization_*.png"
    
    image_files = list(Path(BASE_DIR / model_folder).glob(pattern))
    if image_files:
        return Image.open(image_files[0])
    return None

# App Title
st.title("🔍 Sherlock Holmes Text Generator")
st.markdown("Generate text in the style of Sherlock Holmes using trained MLP language models!")

# Sidebar - Model Selection and Parameters
st.sidebar.header("⚙️ Generation Controls")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model Configuration",
    list(MODELS.keys()),
    help="Choose different combinations of context size, embedding dimension, and activation function"
)

model_config = MODELS[selected_model]

# Display current model info
st.sidebar.info(f"""
**Current Model:**
- Context Length: {model_config['context']}
- Embedding Dim: {model_config['embedding']}
- Activation: {model_config['activation'].upper()}
""")

# Generation parameters
st.sidebar.subheader("Generation Parameters")

num_words = st.sidebar.slider(
    "Number of words to generate",
    min_value=1,
    max_value=100,
    value=20,
    help="How many words should the model generate?"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Lower temperature = more predictable, Higher temperature = more random"
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Set seed for reproducible generation"
)

# Set random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Main content area
tab1, tab2, tab3 = st.tabs(["📝 Text Generation", "📊 Model Info", "🎨 Visualizations"])

# Tab 1: Text Generation
with tab1:
    st.header("Generate Text")
    
    # Input text
    seed_text = st.text_area(
        "Enter seed text (starting context):",
        value="The detective examined the evidence",
        height=100,
        help=f"Provide at least {model_config['context']} words. If you provide fewer, the model will pad with common words."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🎲 Try Random Example", use_container_width=True):
            examples = [
                "Holmes sat in his chair thinking",
                "Watson opened the door and saw",
                "The mystery began on a dark night",
                "The victim was found in the library",
                "A strange letter arrived that morning"
            ]
            seed_text = np.random.choice(examples)
            st.rerun()
    
    if generate_button:
        if not seed_text.strip():
            st.error("Please enter some seed text!")
        else:
            with st.spinner("Loading model and generating text..."):
                # Load model
                model, word_to_idx, idx_to_word = load_model_and_vocab(selected_model)
                
                if model is not None:
                    # Generate text
                    generated_text, oov_words = generate_text(
                        model,
                        word_to_idx,
                        idx_to_word,
                        seed_text,
                        num_words,
                        temperature,
                        model_config['context']
                    )
                    
                    # Display results
                    st.success("✅ Text generated successfully!")
                    
                    # Show OOV warning if any
                    if oov_words:
                        st.warning(f"⚠️ Out-of-vocabulary words found and replaced: {', '.join(set(oov_words))}")
                    
                    # Display generated text
                    st.subheader("Generated Text:")
                    st.markdown(f"```\n{generated_text}\n```")
                    
                    # Highlight the original seed vs generated
                    seed_word_count = len(seed_text.split())
                    generated_word_list = generated_text.split()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Seed Text:**")
                        st.info(' '.join(generated_word_list[:seed_word_count]))
                    
                    with col2:
                        st.markdown("**Generated Continuation:**")
                        st.success(' '.join(generated_word_list[seed_word_count:]))
                    
                    # Generation statistics
                    st.subheader("Generation Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Words", len(generated_word_list))
                    with col2:
                        st.metric("Seed Words", seed_word_count)
                    with col3:
                        st.metric("New Words", len(generated_word_list) - seed_word_count)

# Tab 2: Model Info
with tab2:
    st.header("Model Information")
    
    model_folder = model_config["folder"]
    
    # Display configuration
    st.subheader("Configuration")
    config_df = pd.DataFrame({
        "Parameter": ["Context Length", "Embedding Dimension", "Activation Function"],
        "Value": [model_config['context'], model_config['embedding'], model_config['activation'].upper()]
    })
    st.table(config_df)
    
    # Training summary
    st.subheader("Training Summary")
    summary = load_training_summary(model_folder)
    st.code(summary, language="text")
    
    # Vocabulary info
    with st.spinner("Loading vocabulary information..."):
        _, word_to_idx, _ = load_model_and_vocab(selected_model)
        if word_to_idx:
            st.subheader("Vocabulary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vocabulary Size", len(word_to_idx))
            with col2:
                st.metric("Context Window", model_config['context'])
            
            # Show sample words
            with st.expander("Show Sample Vocabulary (first 50 words)"):
                sample_words = list(word_to_idx.keys())[:50]
                st.write(", ".join(sample_words))

# Tab 3: Visualizations
with tab3:
    st.header("Model Visualizations")
    
    model_folder = model_config["folder"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Training Curve")
        training_img = load_image(model_folder, "training")
        if training_img:
            st.image(training_img, use_column_width=True)
        else:
            st.warning("Training curve not found")
    
    with col2:
        st.subheader("🎨 Embeddings Visualization")
        embedding_img = load_image(model_folder, "embedding")
        if embedding_img:
            st.image(embedding_img, use_column_width=True)
        else:
            st.warning("Embeddings visualization not found")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Generation:
- **Temperature**: Lower values (0.3-0.7) for more coherent text, higher values (1.2-2.0) for more creative/random text
- **Seed Text**: Provide contextually relevant starting text for better results
- **Context Length**: Models with larger context (5) can capture longer dependencies
- **OOV Words**: If your seed contains uncommon words, they'll be replaced with similar words from the vocabulary
""")

st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sherlock Holmes MLP Text Generator | Trained on classic detective stories</p>
</div>

""", unsafe_allow_html=True)
