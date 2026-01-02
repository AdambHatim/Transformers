from spacy.tokens import Doc
import json
import spacy
from collections import Counter

class Tokenizer:
  def __init__(self, train_path, val_path, test_path, load_path = None, translation = "en->de",):

    self.train = self.load_json(train_path)
    self.val = self.load_json(val_path)
    self.test = self.load_json(test_path)

    if load_path is None:
       self.tokenizer = {"SRC": {}, "TGT": {}}
    else:
      with open(load_path, "r", encoding="utf-8") as f:
        self.tokenizer = json.load(f)


    self.train_tokenized = ""
    self.val_tokenized = ""
    self.test_tokenized = ""

    if translation == "en->de":
      self.SRC = "en"
      self.TGT = "de"
      self.spacy_en = spacy.load("en_core_web_sm")
      self.spacy_de = spacy.load("de_core_news_sm")

  def load_json(self, path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        data.append(json.loads(line))
    return data

  def split_text_en(self, text):
    return [tok.text.lower() for tok in self.spacy_en(text)]

  def split_text_de(self, text):
    return [tok.text.lower() for tok in self.spacy_de(text)]

  def split_text(self, text, language):
    if language == "SRC":
      return self.split_text_en(text)
    if language == "TGT":
      return self.split_text_de(text)

  def build_vocabulary(self, min_freq = 1):
    SRC_counter = Counter()
    TGT_counter = Counter()
    for text in self.train:
      SRC_text = self.split_text(text[self.SRC], "SRC")
      TGT_text = self.split_text(text[self.TGT], "TGT")


      SRC_counter.update(SRC_text)
      TGT_counter.update(TGT_text)

    SRC_tokens = []
    TGT_tokens = []

    for tok, freq in SRC_counter.items():
      if freq >= min_freq:
        SRC_tokens.append(tok)

    SRC_tokens.sort()
    SRC_tokens = ["<s>", "</s>", "<pad>", "<unk>"] + SRC_tokens
    for tok, freq in TGT_counter.items():
      if freq >= min_freq:
        TGT_tokens.append(tok)

    TGT_tokens.sort()
    TGT_tokens = ["<s>", "</s>", "<pad>", "<unk>"] + TGT_tokens
    for i, tok in enumerate(SRC_tokens):
      self.tokenizer["SRC"][tok] = i
    for i, tok in enumerate(TGT_tokens):
      self.tokenizer["TGT"][tok] = i

  def tokenize(self, data, language):
    tokenized_data = []
    for text in data:
      tokenized_text = []
      for tok in self.split_text(text, language):
         result = self.tokenizer[language].get(tok, 3)
         tokenized_text.append(result)


      tokenized_text = [0] + tokenized_text + [1]
      tokenized_data.append(tokenized_text)



    return tokenized_data

  def i_tokenize(self, data, language):
      tokenized_data = []
      for text in data:
        tokenized_text = []
        for tok in self.split_text(text, language):
          result = self.tokenizer[language].get(tok, 3)
          tokenized_text.append(result)


        tokenized_text = tokenized_text
        tokenized_data.append(tokenized_text)



      return tokenized_data

  def v_to_k_tokenizer(self):
    self.reverse_tokenizer = {"SRC": {v: k for k, v in self.tokenizer["SRC"].items()},
                              "TGT": {v: k for k, v in self.tokenizer["TGT"].items()}}
  def detokenize_token(self, data, language, is_Token = True):
    detokenized_data = []
    for sentence in data:
      detokenized_sentence = []
      for tok_id in sentence:
        result = self.reverse_tokenizer[language][tok_id]
        detokenized_sentence.append(result)

      detokenized_data.append(detokenized_sentence)

    return detokenized_data

  def detokenize_spacy(self, data, language):
    detokenized_data = []

    nlp = self.spacy_en if language == "SRC" else self.spacy_de

    for sentence in data:
        tokens = [
            self.reverse_tokenizer[language][tok_id]
            for tok_id in sentence
            if self.reverse_tokenizer[language][tok_id] not in ("<s>", "</s>", "<pad>")
        ]

        doc = Doc(nlp.vocab, words=tokens)
        detokenized_data.append(doc.text)

    return detokenized_data
  def pad_sequence(self, tokenized_sentences,device = DEVICE, pad_token=2):
    # find maximum length
    lengths = [len(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
    max_length = max(lengths)
    num_sentences = len(tokenized_sentences)

    batch = torch.zeros((num_sentences, max_length), dtype=torch.long)
    mask = torch.zeros((num_sentences, max_length), dtype=torch.long)
    batch = batch + pad_token

    for i, tokenized_sentence in enumerate(tokenized_sentences):
      t = torch.tensor(tokenized_sentence)
      batch[i, 0:t.size(0)] = t
      mask[i, 0:t.size(0)] = 1

    return batch, mask

  def get_vocab_size(self):
    SRC_vocab_size, TGT_vocab_size = 0, 0
    for k, v in self.tokenizer["SRC"].items():
      SRC_vocab_size += 1
    for k, v in self.tokenizer["TGT"].items():
      TGT_vocab_size += 1

    return SRC_vocab_size, TGT_vocab_size