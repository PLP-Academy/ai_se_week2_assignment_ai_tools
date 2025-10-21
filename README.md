# 🤖 AI Software Engineering Week 3 Assignment

**🎬 Access the project presentation here:**

[![Presentation](https://img.shields.io/badge/🎬%20Open%20The%20Project%20Presentation-blue?style=for-the-badge&logo=github)](https://plp-academy.github.io/ai_se_week2_assignment_ai_tools/)

## 🛠 Technology Stack

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
<img src="https://img.shields.io/badge/spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white" alt="spaCy" />
<img src="https://img.shields.io/badge/Gradio-FF6B35?style=flat-square&logo=gradio&logoColor=white" alt="Gradio" />
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript" />
<img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" alt="HTML5" />
<img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" alt="CSS3" />

## 🌟 Project Overview

Welcome to an exciting journey through the world of **Artificial Intelligence and Software Engineering**! This comprehensive project showcases three fundamental pillars of modern AI development:

- **📊 Classical Machine Learning** with Scikit-learn
- **🧠 Deep Learning** with TensorFlow
- **📝 Natural Language Processing** with spaCy

Each component demonstrates real-world applications, from classifying iris flowers to recognizing handwritten digits and analyzing sentiment in text.

---

## 🎯 Main Tasks & Approaches

### 1. **Classical ML with Scikit-learn** 🌸
**Task:** Iris Flower Classification

**Approach:**
- **Dataset:** Used the famous Iris dataset (150 samples, 4 features, 3 species)
- **Algorithm:** Decision Tree Classifier for interpretable results
- **Preprocessing:** Standard scaling and missing value imputation
- **Evaluation:** Accuracy, precision, recall, and comprehensive classification reports

**Key Features:**
- Robust data preprocessing pipeline
- Reproducible train/test splits (42 random state)
- Multi-class classification metrics
- Educational focus on ML fundamentals

---

### 2. **Deep Learning with TensorFlow** 🔢
**Task:** MNIST Handwritten Digit Recognition

**Approach:**
- **Architecture:** Custom Convolutional Neural Network (CNN)
- **Layers:** Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout
- **Training:** 15 epochs with data augmentation for robustness
- **Achievement:** 99.42% test accuracy

**Key Features:**
- **Data Augmentation:** Rotation, shift, zoom for diverse handwriting styles
- **Dropout Regularization:** Prevents overfitting (50% dropout rate)
- **Model Persistence:** Saved trained model for deployment
- **Visualization:** Training accuracy graphs and sample predictions

---

### 3. **Natural Language Processing with spaCy** 📖
**Task:** Amazon Product Reviews Analysis

**Approach:**
- **NLP Pipeline:** Named Entity Recognition (NER) and sentiment analysis
- **Entities:** Product names, brands, organizations
- **Sentiment:** Rule-based classification using keyword matching
- **Visualization:** spaCy's displaCy for entity highlighting

**Key Features:**
- **Pretrained Models:** spaCy's `en_core_web_sm` for robust tokenization
- **Custom Sentiment Rules:** Positive/negative keyword dictionaries
- **Entity Extraction:** Real-time processing of review text
- **Interactive Visualization:** Beautiful entity highlighting

---

## 🚀 Bonus Features

### **Interactive Presentation** 🎬
- **Automated Slideshow:** Auto-advancing presentation with voice narration
- **Gemini AI Integration:** Text enhancement for more natural speech
- **Voice Control:** Pause/resume functionality with visual indicators
- **Responsive Design:** Works perfectly on desktop and mobile

### **Gradio Deployment** 🌐
- **Live Demo:** Web interface for MNIST digit classification
- **Real-time Prediction:** Draw digits and get instant results
- **Model Integration:** Uses the trained CNN for accurate predictions
- **User-friendly UI:** Simple sketchpad interface

---

## 🛠 Technical Architecture

### **Project Structure**
```
📁 AI_SE_Week3_Assignment/
├── 📁 practical/
│   ├── 📁 scikit-learn/iris_decision_tree.ipynb
│   ├── 📁 tensorflow/mnist_cnn.ipynb
│   └── 📁 spacy/nlp_spacy.ipynb
├── 📁 bonus/
│   ├── app.py (Gradio interface)
│   └── mnist_cnn_improved_model.h5
├── 📁 docs/
│   ├── presentation.html (Interactive slideshow)
│   ├── env-loader.js (API key management)
│   └── env.json (Secure configuration)
└── 📋 README.md (This file)
```

### **Technology Stack**
- **Python:** Core ML implementation
- **TensorFlow:** Deep learning framework
- **Scikit-learn:** Classical ML algorithms
- **spaCy:** NLP processing
- **Gradio:** Web deployment
- **HTML/CSS/JavaScript:** Interactive presentation

---

## 🎨 User Experience Flow

### **For Learners** 📚
1. **Study Theory:** Understand ML/DL/NLP concepts
2. **Run Notebooks:** Execute practical examples
3. **View Presentation:** Interactive slideshow with narration
4. **Try Gradio App:** Test digit recognition live

### **For Developers** 💻
1. **Review Code:** Examine implementation details
2. **Understand Architecture:** Study design patterns
3. **Deploy Locally:** Run Gradio app and presentation
4. **Extend Features:** Add new models or capabilities

---

## 🌟 Key Learning Outcomes

### **Technical Skills**
- ✅ **ML Pipeline Design:** Data preprocessing to deployment
- ✅ **Model Selection:** Choosing appropriate algorithms
- ✅ **Performance Optimization:** Achieving 99.42% accuracy
- ✅ **Web Integration:** Gradio and interactive presentations

### **Soft Skills**
- ✅ **Problem Solving:** Debugging and optimization
- ✅ **Documentation:** Clear code comments and explanations
- ✅ **Presentation:** Engaging technical communication
- ✅ **Ethics:** Bias mitigation and responsible AI

---

## 🚀 Getting Started

### **Quick Start**
1. **Clone Repository:** `git clone https://github.com/PLP-Academy/ai_se_week2_assignment_ai_tools.git`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Notebooks:** Execute `.ipynb` files in Jupyter
4. **View Presentation:** Open `docs/presentation.html` in browser
5. **Try Gradio App:** `python bonus/app.py`

### **Prerequisites**
- Python 3.8+
- Jupyter Notebook
- TensorFlow, Scikit-learn, spaCy
- Modern web browser

---

## 📊 Performance Highlights

| Task | Model | Accuracy | Key Achievement |
|------|-------|----------|-----------------|
| **Iris Classification** | Decision Tree | 100% | Perfect classification |
| **MNIST Recognition** | CNN | 99.42% | State-of-the-art results |
| **Sentiment Analysis** | Rule-based | 85%+ | Effective text analysis |

---

## 🎯 Project Impact

This assignment demonstrates the complete **AI development lifecycle**:

1. **🔍 Problem Identification:** Real-world ML challenges
2. **📊 Data Processing:** Cleaning and preparation
3. **🤖 Model Development:** Training and optimization
4. **✅ Evaluation:** Rigorous performance testing
5. **🌐 Deployment:** Web interfaces and presentations
6. **📚 Documentation:** Comprehensive explanations

---

## 🔮 Future Enhancements

- **Multi-digit Recognition:** Extend MNIST for number sequences
- **Advanced NLP:** Transformer-based sentiment analysis
- **Cloud Deployment:** Online Gradio interface
- **Mobile App:** React Native digit classifier
- **Real-time Video:** Live handwriting recognition

---

## 🙏 Acknowledgments

**Author:** George Wanjohi
**Institution:** PLP Academy
**Focus:** AI Software Engineering Excellence

*"Bridging the gap between theoretical AI concepts and practical, deployable solutions"*

---

## 📞 Contact & Resources

- **📧 Email:** george.wanjohi@plpacademy.ac.ke
- **🌐 Presentation:** [View Interactive Slideshow](https://plp-academy.github.io/ai_se_week2_assignment_ai_tools/) (opens in new tab)
- **💻 Repository:** [GitHub Source Code](https://github.com/PLP-Academy/ai_se_week2_assignment_ai_tools.git)
- **📖 Documentation:** Comprehensive inline comments and docstrings

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/PLP-Academy/ai_se_week2_assignment_ai_tools.svg?style=social&label=Star)](https://github.com/PLP-Academy/ai_se_week2_assignment_ai_tools)

</div>