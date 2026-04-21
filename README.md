# 🍏 Apple Sentiment AI

> **Can a computer understand how people *feel* about Apple products?**
> This project does exactly that — it reads tweets and reviews about Apple and tells you whether the opinion is **Positive 🟢**, **Negative 🔴**, or **Neutral ⚪**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/Model-XLM--RoBERTa-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🤔 What does this app do?

Imagine you type:

> *"iPhone 16 Pro Max camera is absolutely insane 🔥"*

The app will:
1. **Detect** that this is about the `iPhone 16 Pro Max` and its `Camera`
2. **Analyse** the emotion in the sentence
3. **Tell you** → **Positive 🟢 (80.2% confident)**

It works with **English, French, Spanish** and even **Emojis!**

---

## ✨ Features

| Feature | What it does |
|---|---|
| 🎯 **Aspect Detection** | Identifies *which* Apple product is being talked about |
| 🧠 **AI Sentiment Analysis** | Uses a state-of-the-art multilingual AI model |
| 😀 **Emoji Decoding** | Understands the emotion behind emojis like 😡 or 🔥 |
| 🏷️ **Named Entity Recognition** | Spots real-world names and places in the text |
| 📊 **Confidence Scores** | Shows *how sure* the AI is about its answer |
| 📈 **N-Gram Analysis** | Finds common word pairs and triplets |
| 🔤 **NLP Pipeline** | Shows every step of how the text is cleaned and processed |

---

## 📱 Apple Products Supported (2024–2025)

The app knows about **40+ Apple products and features**, including:

- **iPhone 16 family** — iPhone 16, 16 Plus, 16 Pro, 16 Pro Max
- **Mac** — MacBook Air (M3), MacBook Pro (M4), iMac (M4), Mac Mini (M4), Mac Studio
- **iPad** — iPad Pro (M4), iPad Air (M3), iPad Mini
- **Apple Watch** — Series 10, Ultra 2
- **AirPods** — AirPods Pro 2, AirPods Max (USB-C)
- **Apple Vision Pro** + visionOS
- **Software** — iOS 18, macOS Sequoia, Apple Intelligence, Siri
- **Chips** — M4, M3, A18 Pro, A18 Bionic
- **Features** — Dynamic Island, Action Button, Face ID, Ceramic Shield, USB-C

---

## 📂 Dataset

The model was trained and tested using real Apple-related tweets from Kaggle:

🔗 **[Apple Tweets Sentiment Dataset](https://www.kaggle.com/datasets/anishdabhane/apple-tweets-sentiment-dataset)** — by *anishdabhane*

---

## 🤖 The AI Model

This app uses **XLM-RoBERTa** — a large language model created by [Cardiff NLP](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).

> Think of it like a very well-read robot that has studied millions of tweets and learned to understand whether they sound happy, sad, or neutral — in multiple languages.

---

## 🖥️ How to Run It Yourself

### Step 1 — Make sure Python is installed
Download Python from [python.org](https://www.python.org/downloads/) if you don't have it.
You can check by opening Terminal and typing:
```
python3 --version
```

### Step 2 — Download this project
Click the green **Code** button on this page → **Download ZIP** → unzip it somewhere on your computer.

Or if you use Git:
```bash
git clone https://github.com/YOUR_USERNAME/apple-sentiment-ai.git
cd apple-sentiment-ai
```

### Step 3 — Install the required packages
Open Terminal, go into the project folder and run:
```bash
pip3 install -r requirements.txt
```

> ⏳ This may take a few minutes — it's downloading the AI libraries.

### Step 4 — Run the app
```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501` and you'll see the app! 🎉

> 💡 **First launch takes 1–3 minutes** because it downloads the AI model (~1 GB). Every run after that is instant.

---

## 📦 Requirements

```
streamlit
pandas
nltk
emoji
torch
transformers
```

> All of these are installed automatically in Step 3 above.

---

## 🗂️ Project Structure

```
ML_Mini Project/
│
├── app.py          ← The entire app (all the code lives here)
├── requirements.txt← List of packages needed to run
└── README.md       ← This file
```

---

## 🧰 How the NLP Pipeline Works

Don't worry if you don't know what NLP means — here's a simple explanation of what happens when you type something:

```
You type →  "iPhone 16 Pro Max camera is absolutely insane 🔥"
             ↓
Step 1: Expand contractions   → "can't" becomes "cannot"
Step 2: Decode emojis         → 🔥 is recognised as a positive signal
Step 3: Remove noise          → URLs, @mentions removed
Step 4: Tokenise              → Split into individual words
Step 5: POS Tagging           → Label each word (noun, verb, adjective…)
Step 6: Lemmatise             → "running" → "run", "cameras" → "camera"
Step 7: Remove stopwords      → Remove "is", "the", "a", etc.
Step 8: AI Model              → XLM-RoBERTa predicts sentiment
             ↓
Result → "Positive 🟢 — 80.2% confident"
```

---

## 💡 Example Inputs to Try

| Input | Expected Result |
|---|---|
| `iPhone 16 Pro Max camera is absolutely insane 🔥` | Positive 🟢 |
| `Apple Vision Pro is way too expensive 😤` | Negative 🔴 |
| `AAPL stock dropped after Tim Cook's event 📉` | Negative 🔴 |
| `Le nouvel iPad Pro M4 est incroyable!` | Positive 🟢 |
| `AirPods Max battery life still mediocre 🙄` | Negative/Neutral |

---

## 👨‍💻 Built With

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | The web app framework |
| [HuggingFace Transformers](https://huggingface.co) | Loading and running the AI model |
| [NLTK](https://www.nltk.org) | Natural Language Processing (tokenising, POS tagging, NER) |
| [XLM-RoBERTa](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | The multilingual sentiment AI model |
| [PyTorch](https://pytorch.org) | Powers the AI model underneath |
| [Pandas](https://pandas.pydata.org) | Data handling |
| [emoji](https://pypi.org/project/emoji/) | Emoji detection and decoding |

---

## ❓ Common Questions

**Q: Do I need an internet connection?**
Yes — only for the first run to download the AI model. After that, it works offline.

**Q: Is this free?**
Yes, 100% free and open source.

**Q: My terminal says "command not found: streamlit" — what do I do?**
Try running: `python3 -m streamlit run app.py` instead.

**Q: The app is slow on first load — is that normal?**
Yes! The AI model file is about 1 GB. After the first download it's cached and loads quickly every time.

**Q: Does it work on Windows?**
Yes — just use Command Prompt or PowerShell instead of Terminal.

---

## 📄 License

This project is open source under the [MIT License](LICENSE) — feel free to use, modify, and share it.

---

<div align="center">
  Made with ❤️ · Powered by XLM-RoBERTa · Built with Streamlit
</div>
