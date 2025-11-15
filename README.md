# 🛡️ DomesticSafetyNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonkrishna57/DomesticSafetyNet/blob/main/DomesticSafetyNet.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered system that **anonymizes text** and **classifies domestic safety risk levels** using a fine-tuned DistilBERT model. This project prioritizes privacy protection while assessing potential threat indicators in text messages.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Privacy & Anonymization](#privacy--anonymization)
- [Risk Level Classifications](#risk-level-classifications)
- [Web Interface](#web-interface)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

DomesticSafetyNet is designed to help identify potential domestic safety threats while maintaining user privacy. The system automatically removes personally identifiable information (PII) from text before analyzing its risk level, making it suitable for sensitive applications where privacy is paramount.

The project combines:
- **Advanced text anonymization** using regex patterns and Named Entity Recognition (NER)
- **Deep learning classification** with a fine-tuned DistilBERT model
- **Interactive web interface** built with Gradio

## ✨ Features

- **🔒 Privacy-First Design**: Automatically anonymizes PII before processing
- **🎯 Multi-Level Risk Classification**: Categorizes text into four risk levels (Low, Medium, High-Urgency, Immediate-Threat)
- **🤖 Transformer-Based Model**: Utilizes fine-tuned DistilBERT for accurate classification
- **🌐 Interactive Web UI**: User-friendly Gradio interface for real-time predictions
- **📊 Confidence Scoring**: Provides confidence levels for each prediction
- **📝 Prediction History**: Tracks recent predictions for analysis

## 🛠️ Technology Stack

- **Python 3.x**
- **TensorFlow** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained language models
- **spaCy** - Natural Language Processing for NER
- **Gradio** - Web interface creation
- **pandas** - Data manipulation
- **scikit-learn** - Model evaluation metrics
- **NumPy** - Numerical operations

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) GPU for faster training

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/menonkrishna57/DomesticSafetyNet.git
   cd DomesticSafetyNet
   ```

2. **Install required packages:**
   ```bash
   pip install tensorflow transformers pandas scikit-learn spacy gradio
   ```

3. **Download spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook DomesticSafetyNet.ipynb
   ```

   Or run directly in [Google Colab](https://colab.research.google.com/github/menonkrishna57/DomesticSafetyNet/blob/main/DomesticSafetyNet.ipynb) (recommended for GPU access).

## 🚀 Usage

### Training the Model

1. Load the dataset (`synthetic_data_1000.json`)
2. Run the anonymization preprocessing
3. Fine-tune the DistilBERT model (approximately 15 epochs)
4. Evaluate model performance on test data

### Making Predictions

```python
# Example prediction
new_sentence = "He is threatening me again and following my car."

# Anonymize and clean the text
cleaned_sentence = anonymize_and_clean_for_bert(new_sentence)

# Get prediction
inputs = tokenizer(cleaned_sentence, return_tensors="tf", truncation=True, padding=True, max_length=128)
outputs = loaded_model(inputs)
predicted_label = id_to_label[tf.argmax(outputs.logits, axis=-1).numpy()[0]]

print(f"Predicted Risk Level: {predicted_label}")
```

### Using the Web Interface

Run the Gradio interface cell in the notebook to launch an interactive web UI where you can:
- Enter text for analysis
- View anonymized versions of inputs
- See risk level predictions with confidence scores
- Track prediction history

## 📊 Dataset

The project uses a synthetic dataset (`synthetic_data_1000.json`) containing:
- **999 text samples** with associated risk levels
- **Four risk categories**: Low, Medium, High-Urgency, Immediate-Threat
- Diverse scenarios representing various domestic safety situations

Dataset structure:
```json
{
  "text": "Example text content...",
  "labels": {
    "risk_level": "Immediate-Threat"
  }
}
```

## 🧠 Model Architecture

### Base Model
- **DistilBERT** (distilbert-base-uncased)
- Lightweight version of BERT with 97% performance retention
- Faster inference time while maintaining accuracy

### Fine-Tuning Details
- **Optimizer**: Adam (learning rate: 5e-5)
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 15
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Train/Test Split**: 80/20 with stratification

## 🔐 Privacy & Anonymization

The system anonymizes the following PII before processing:

| PII Type | Pattern | Replacement Token |
|----------|---------|-------------------|
| Phone Numbers | Indian/International formats | `[PHONE]` |
| Email Addresses | Standard email format | `[EMAIL]` |
| Aadhaar Numbers | 12-digit format | `[AADHAAR]` |
| PAN Card | Indian PAN format | `[PAN]` |
| License Plates | Vehicle registration | `[LICENSE_PLATE]` |
| Addresses | Street addresses | `[ADDRESS]` |
| Person Names | NER-detected | `[PERSON]` |
| Organizations | NER-detected | `[ORG]` |
| Locations | NER-detected | `[GPE]`/`[LOC]` |

## 📈 Risk Level Classifications

The model classifies text into four categories:

| Risk Level | Description | Color Code |
|------------|-------------|------------|
| **Low** | Minimal or no threat detected | 🟢 Green |
| **Medium** | Moderate concern, monitoring advised | 🟡 Yellow |
| **High-Urgency** | Serious threat, immediate attention needed | 🟠 Orange |
| **Immediate-Threat** | Critical danger, urgent intervention required | 🔴 Red |

## 🌐 Web Interface

The Gradio-based interface provides:
- **Real-time analysis** of input text
- **Visual confidence indicators** with color-coded progress bars
- **Anonymized text display** showing privacy protection
- **Prediction history** tracking last 5 analyses
- **Share functionality** for collaborative use

## 📊 Results

The model achieves high accuracy in classifying risk levels while maintaining privacy. Detailed evaluation metrics including precision, recall, and F1-scores are provided in the classification report within the notebook.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Improving anonymization patterns
- Adding support for more languages
- Enhancing model accuracy
- Expanding the dataset
- UI/UX improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Krishna Menon

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **spaCy** for NER capabilities
- **Google Colab** for providing free GPU resources
- **Gradio** for the intuitive web interface framework

## 📧 Contact

Krishna Menon - [@menonkrishna57](https://github.com/menonkrishna57)

Project Link: [https://github.com/menonkrishna57/DomesticSafetyNet](https://github.com/menonkrishna57/DomesticSafetyNet)

---

**⚠️ Important Note**: This system is designed as a support tool and should not be the sole basis for critical safety decisions. Always consult with appropriate professionals for serious safety concerns.
