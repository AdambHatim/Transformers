# Transformer EN → DE (PyTorch)

This project is a **from-scratch implementation of a Transformer model** for **English → German machine translation**, inspired by *Attention Is All You Need*.

It includes:
- Full Transformer architecture (encoder / decoder)
- spaCy-based tokenizer
- Training with teacher forcing
- **Greedy decoding** and **beam search**
- Evaluation using **BLEU (sacrebleu)**

---

## 📂 Project Structure

```
.
├── models.py              # Transformer architecture (from scratch)
├── Tokenizer.py           # spaCy tokenizer + vocabulary
├── Transformers.ipynb     # Training, evaluation, decoding
├── Dataset/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── tokenizer.json
├── transformer_checkpoint.pth   # ignored by git
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch spacy sacrebleu matplotlib numpy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## 📊 Dataset

Data is stored in **JSONL** format:

```json
{"en": "i eat meat", "de": "ich esse fleisch"}
```

Paths are configured in `Transformers.ipynb`.

---

## 🧠 Model

- Embedding size: 512  
- Attention heads: 8  
- Encoder layers: 3  
- Decoder layers: 3  
- FFN dimension: 2048  
- Dropout: 0.1  

The Transformer is fully implemented in `models.py` without using `torch.nn.Transformer`.

---

## 🚀 Training

- Optimizer: Adam (`lr = 1e-4`)
- Loss: CrossEntropyLoss (padding ignored)
- Checkpoint saved after each epoch

```python
torch.save({...}, "transformer_checkpoint.pth")
```

The checkpoint file is ignored via `.gitignore`.

---

## 📈 Results

Validation BLEU scores:

- **Greedy decoding**: ~25  
- **Beam search (beam=5)**: ~30  

---

## 🔍 Decoding

Implemented decoding strategies:
- Greedy decoding
- Beam search decoding

Evaluation is done using **sacrebleu**.

---

## 📝 Notes

- Educational, research-oriented implementation
- Focus on clarity and understanding of the Transformer architecture
- Suitable as a learning or portfolio project

---

## 👤 Author

Project developed for educational purposes.
