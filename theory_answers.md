# Part 1 — Theoretical Answers

## Q1: TensorFlow vs PyTorch (short)
**TensorFlow vs PyTorch — key differences**
- **Execution model:** PyTorch uses dynamic (eager) execution which is Pythonic and easy to debug. TensorFlow historically used static graphs but TensorFlow 2.x defaults to eager execution and is closer to PyTorch.
- **Use cases:** PyTorch is commonly used for research and fast prototyping due to its intuitive API. TensorFlow is well-suited for production and deployment (TF Serving, TF Lite, TF Hub).
- **Ecosystem & tooling:** TensorFlow has a broad production ecosystem; PyTorch has strong research adoption and growing production tools (TorchServe).
- **Rule of thumb:** choose PyTorch for experimentation/research; choose TensorFlow when you need mature production tooling or specific TF integrations.

## Q2: Two use cases for Jupyter Notebooks
1. **Interactive prototyping and experiments:** run small code blocks iteratively, inspect outputs and tweak models without running a full script.  
2. **Reproducible reports and visualizations:** combine narrative, code, plots, and results in one document for sharing and teaching.

## Q3: How spaCy improves NLP vs basic string ops
- **Tokenization & linguistics:** spaCy provides robust, language-aware tokenization, POS tagging, and dependency parsing; basic string ops cannot reliably split or normalize text.
- **Pretrained models & NER:** spaCy includes pretrained pipelines for Named Entity Recognition (PRODUCT, ORG, PERSON), which work across varied text; string searches are brittle and miss variations.
- **Pipeline & extensibility:** spaCy offers matchers and rule-based add-ons (EntityRuler) to incorporate custom patterns without reinventing low-level parsing.

## Comparative Analysis: Scikit-learn vs TensorFlow

| Aspect | Scikit-learn | TensorFlow |
|--------|--------------|------------|
| Target applications | Classical ML: SVM, trees, clustering, preprocessing | Deep learning: neural networks, CNNs, RNNs, production ML pipelines |
| Ease for beginners | Very beginner-friendly, consistent API | Steeper learning curve (TF 2.x is easier), more concepts |
| Community & Ecosystem | Mature for classical ML; many utilities | Massive (Google-backed), strong production tooling and model-serving options |

**Short summary:** Use Scikit-learn for classical ML tasks and quick baselines. Use TensorFlow when needing deep neural networks or deployment tools; PyTorch is an excellent alternative for research-focused deep learning.

# Part 3 — Ethics & Optimization

## Ethical Considerations

### MNIST Model Bias
The MNIST dataset, while widely used, primarily consists of digits written by a specific demographic (NIST employees and high school students). This can lead to **representation bias** if the model is deployed in a real-world scenario where handwritten digits come from a more diverse population (e.g., different writing styles, ages, or cultural backgrounds). The model might perform poorly on digits that deviate significantly from its training data, leading to unfair or inaccurate classifications for certain groups.

**Mitigation Strategies:**
- **Dataset Diversity:** Collect and augment training data to include a wider variety of handwritten styles from diverse demographics.
- **Fairness Metrics:** Implement fairness metrics (e.g., equal accuracy across different demographic groups if such labels were available) during evaluation to detect and address disparities.

### Amazon Review Model Bias
The rule-based sentiment analysis model for Amazon reviews relies on predefined lists of positive and negative words. This approach is prone to **lexical bias** and **contextual bias**.
- **Lexical Bias:** The chosen words might not capture the full spectrum of sentiment or could be culturally biased. For example, a word considered positive in one context might be neutral or negative in another.
- **Contextual Bias:** The model lacks understanding of sarcasm, irony, or nuanced language. A review like "This product is *so* good, I'm *so* disappointed it broke" would likely be misclassified as positive due to the presence of "good." This can lead to inaccurate sentiment analysis, potentially misrepresenting customer feedback.

**Mitigation Strategies:**
- **Better Rule Design:** Expand and refine keyword lists, incorporating more nuanced terms and considering multi-word expressions.
- **Contextual NLP Models:** For more robust sentiment analysis, transition to machine learning models (e.g., BERT-based models) that can understand context and semantics, rather than just keyword presence.
- **Human Review:** Periodically review model classifications with human annotators to identify and correct biases in the rule set.

## Troubleshooting Challenge

### Buggy TensorFlow Code
Here's a simple example of a buggy TensorFlow CNN that might encounter issues due to an incorrect input shape for the first convolutional layer. If the input data is not correctly reshaped to match `(height, width, channels)`, the model will fail during compilation or the first forward pass.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Simulate incorrect input shape (e.g., missing channel dimension)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# BUG: Not reshaping x_train and x_test to include the channel dimension (28, 28, 1)

model_buggy = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Expects 3D input (H, W, C)
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model_buggy.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# This line would cause an error because x_train has shape (60000, 28, 28) not (60000, 28, 28, 1)
# model_buggy.fit(x_train, y_train, epochs=1)
```

### Fixed TensorFlow Code
The corrected version ensures that the input data `x_train` and `x_test` are properly reshaped to include the channel dimension `(batch_size, height, width, channels)`, which is `(None, 28, 28, 1)` for grayscale images, before being fed into the `Conv2D` layer.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess data correctly
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# FIX: Reshape x_train and x_test to include the channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model_fixed = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model_fixed.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# This will now run without error
# model_fixed.fit(x_train, y_train, epochs=1)
```

### Concluding Note on Ethical AI
Addressing ethical considerations and potential biases is as crucial as achieving high performance in AI model development. Proactive identification and mitigation of biases, coupled with robust troubleshooting practices, ensure that AI systems are not only efficient but also fair, reliable, and responsible in their real-world applications.
