## Comparative Study: Sherlock Holmes (Natural Language) vs Linux Kernel (Structured Language)
* Models were too large too push(faced https timeout error while pushing...tried pushing thrice, I can show you the .pt and .pth locally and run the streamlit app locally)
### **1. Dataset Characteristics**

| Aspect                     | Sherlock Holmes (Category I)                                             | Linux Kernel (Category II)                                                       |
| -------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Nature of Text**         | Natural, narrative English text from literature.                         | Highly structured programming language (C code/comments).                        |
| **Vocabulary Size**        | Large and diverse – includes verbs, adjectives, nouns, and punctuation.  | Smaller but more repetitive – includes syntax tokens, keywords, and identifiers. |
| **Context Predictability** | Low – next-word prediction depends on long, abstract semantic relations. | High within local scope – predictable patterns like `if`, `for`, `return`, etc.  |
| **Sentence Structure**     | Flexible and creative with variable-length sentences.                    | Strict syntax rules with limited variability.                                    |

**Insight:**
Natural language exhibits rich variability and subtle semantic dependencies, making it harder for smaller models to capture long-range dependencies.
In contrast, structured code follows strict grammar and repetitive constructs, which simplifies local context prediction but demands precise token-level modeling.

---

### **2. Model Performance and Loss Curves**

#### **Training vs Validation Loss**

* **Sherlock Holmes models** showed **rapid overfitting**:

  * Training loss dropped steadily and reached very low values.
  * Validation loss continuously increased, indicating the model memorized patterns instead of generalizing.
* **Linux Kernel models** displayed **less severe overfitting but poor generalization**:

  * Training loss dropped to nearly zero.
  * Validation loss also increased, but the rate was more gradual.
  * Suggests the model captured structural patterns (syntax) but failed to generalize to unseen contexts.

#### **Validation Accuracy Trends**

* **Sherlock Holmes:** Validation accuracy plateaued around **10%**, with little improvement across configurations.
* **Linux Kernel:** Validation accuracy initially peaked (~23%) but **declined steadily**, reflecting overfitting after early epochs.

#### **Effect of Hyperparameters**

| Hyperparameter                         | Observation                                                                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Context Size (3 → 5)**               | Slight improvement in capturing structure for both datasets, but minimal gain for natural text due to long-range dependencies. |
| **Embedding Dimension (32 → 64)**      | Marginally improved learning capacity but also accelerated overfitting.                                                        |
| **Activation Function (ReLU vs Tanh)** | ReLU trained faster but overfit more severely; Tanh gave smoother convergence and slightly better validation stability.        |

---

### **3. Qualitative Text Generations**

* **Sherlock Holmes:**
  Generated sequences often followed grammatical structure but lacked semantic coherence. Sentences sounded syntactically correct but contextually random (e.g., “the detective upon the room but it not”).
  → Indicative of **semantic uncertainty** and limited contextual recall.

* **Linux Kernel:**
  Generated code snippets mimicked syntax (e.g., “if (ptr == NULL) return;”), maintaining parentheses and keywords correctly. However, variable names and logic flow were nonsensical.
  → Indicative of **surface-level pattern learning** without functional understanding.

---

### **4. Embedding Visualizations (t-SNE / PCA)**

* **Sherlock Holmes embeddings:**
  Showed **semantic clustering** of similar words (e.g., pronouns, verbs) but with noisy boundaries due to ambiguous natural context.

* **Linux Kernel embeddings:**
  Displayed **clear structural clusters**, grouping keywords (`if`, `for`, `return`) separately from identifiers and symbols.
  → Suggests stronger syntactic consistency in code compared to semantic fluidity in natural language.

---

### **5. Insights on Learnability**

| Aspect                   | Natural Language (Sherlock Holmes)             | Structured Language (Linux Kernel)                                      |
| ------------------------ | ---------------------------------------------- | ----------------------------------------------------------------------- |
| **Predictability**       | Low – context-dependent, semantically rich.    | High – syntax-driven and locally consistent.                            |
| **Overfitting Behavior** | Strong – memorizes training text easily.       | Moderate – captures syntax but fails to generalize logic.               |
| **Embedding Structure**  | Semantically dispersed clusters.               | Compact, rule-based clusters.                                           |
| **Model Difficulty**     | Requires larger context and semantic modeling. | Easier to learn surface patterns, but understanding logic remains hard. |

**Conclusion:**
Natural language is **harder to learn** due to its semantic ambiguity and long-range dependencies, demanding higher model capacity and context awareness.
Structured code, though more repetitive and rule-bound, still challenges models in capturing **logical dependencies** rather than surface syntax.
Overall, MLPs with small context windows can mimic syntax but struggle with deeper understanding in both domains — highlighting the need for sequence-aware architectures (like RNNs or Transformers) for meaningful text generation.
