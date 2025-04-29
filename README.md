# 📸 blip-pytorch-captioning

> **From Pixels to Captions — Building an Intelligent Image Captioning System with PyTorch and BLIP**

---

### 🛠 Project Overview

This repository combines the power of state-of-the-art deep learning models with real-world social media data to create a robust **image captioning system**.

We leverage:
- **Instagram Image Caption Dataset** 📷  
  [Instagram Images with Captions - Kaggle Dataset](https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions?select=instagram_data2)
- **BLIP (Bootstrapped Language Image Pretraining) Model** 🧠  
  [Salesforce BLIP Large Model on HuggingFace](https://huggingface.co/Salesforce/blip-image-captioning-large)
- **PyTorch** ⚙️  
  to orchestrate the entire training, evaluation, and fine-tuning pipeline.

---

### 🔥 Key Highlights
- Efficiently pre-process and load Instagram images and captions.
- Fine-tune the **BLIP Large** model using **PyTorch**.
- Implement custom training and evaluation loops with metric tracking (BLEU, ROUGE).
- Support for 5-fold cross-validation and automatic best model saving.
- Mixed precision training (`torch.cuda.amp`) for optimal GPU memory usage.
- **Flexible Script Architecture**:
  - 📓 **Jupyter Notebook Cell Scripts**:  
    Modular scripts designed for easy cell-by-cell execution and debugging.  
    Available inside the folder: `scripts/Jupyter Notebook Cell Scripts`
  - 💻 **VS Code Scripts**:  
    Complete runnable `.py` scripts for seamless execution on VS Code or terminal.  
    Available inside the folder: `scripts/VS Code Scripts`

---

### 📈 Project Goal
To build an end-to-end image-to-text captioning system that can generate natural, engaging captions for unseen Instagram images, pushing the boundaries of visual storytelling.

---

### 🚀 Technologies Used
- PyTorch 🐍
- Hugging Face Transformers 🤗
- Kaggle Datasets 📂
- NLTK, ROUGE Metrics 📏
- CUDA Acceleration ⚡

---

# ✨ Let's turn pixels into poetry!

---
