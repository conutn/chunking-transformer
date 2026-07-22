"""
Vocabulary loading, tokenizer, and dataset classes used by
train.py.

The tokenizer expects a vocabulary file with one token per line,
continuation pieces prefixed with "##". Continuation pieces are re-prefixed
with "§" so the tokenizer can distinguish between the two types.
"""

import unidecode
import torch
from torch.utils.data import Dataset


def read_vocab(fname):
    file = open(fname, encoding="utf-8").read().splitlines()
    vocab = []
    for word in file:
        if any((ord(ch) > 128) for ch in word):
            continue
        if word[:2] == "##":
            vocab.append("§" + word[2:])
        else:
            vocab.append(word)
    vocab = list(filter(lambda x: len(x) <= 4 or ("§" in x and len(x) == 5), vocab))
    vocab += ["[SEP]", "[PAD]", "[UNK]"]
    return vocab


def get_vocab_dct_encoder(vocab):
    vocab_dct = {vocab[idx]: idx for idx in range(len(vocab))}
    return vocab_dct


whitespace = {
    "\u0009",
    "\u000a",
    "\u000b",
    "\u000c",
    "\u000d",
    "\u0020",
    "\u0085",
    "\u00a0",
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
}


def is_whitespace(char: str) -> bool:
    """Returns true if char is a whitespace character."""
    return char in whitespace


class Tokenizer:
    # taken from Hugging Face's tokenizer implementation
    """A utility class for tokenizing text."""

    def clean_text(self, text: str, is_wiki=False) -> str:
        """Replaces non-alphanumeric characters with an ASCII
        approximation.
        Can be used with raw Wikipedia article text; removes all
        text after the References section in the article."""

        text = unidecode.unidecode(text)
        text = text.lower()
        output = []

        for sentence in text.splitlines():
            if sentence in {
                "references",
                "further reading",
                "sources",
                "external links",
            }:
                break
            if is_wiki and len(sentence) and sentence[-1] not in self.punc:
                continue
            for char in sentence:
                if is_whitespace(char):
                    output.append(" ")
                else:
                    output.append(char)
            output.append(" ")

        return "".join(output)

    def __init__(self, vocab: list, unk_token: str):
        self.vocab = vocab
        self.unk_token = unk_token
        self.punc = {".", "!", "?"}  # for separating sentences

    def tokenize(self, text: str, is_wiki=False) -> str:
        """Converts text into tokens.
        Suffixes are prepended with a section character (§)."""

        set_vocab = set(self.vocab)

        # clean text and strip whitespace
        text = self.clean_text(text, is_wiki=is_wiki)
        text = text.strip()
        tokens = text.split()

        output = []

        for token in tokens:
            chars = list(token)

            is_valid = True
            start = 0
            sub_tokens = []

            # take tokens greedily from the beginning of the word
            while start < len(chars):
                end = len(chars)
                cur_substr = None

                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "§" + substr
                    if substr in set_vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr == None:
                    is_valid = False
                    break

                sub_tokens.append(cur_substr)
                start = end

            if not is_valid:
                output.append(self.unk_token)
            else:
                output.extend(sub_tokens)

        return output


class ChunkedDataset(Dataset):
    """Next-token-prediction dataset: contiguous windows of `block_size`
    tokens, with targets shifted by one position."""

    def __init__(self, tokens, block_size=256):
        self.ids = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.block_size]
        y = self.ids[idx + 1 : idx + 1 + self.block_size]
        return x, y

TransformerDataset = ChunkedDataset


def get_split(data, splits=[0.98, 0.01, 0.01]):
    """Splits a flat token list into train/valid/test according to `splits`
    (fractions of the total, applied in order)."""
    bounds = [0]
    for i in splits:
        bounds.append(bounds[-1] + int(len(data) * i))
    bounds[-1] = -1
    data_splits = []
    for i in range(len(bounds) - 1):
        data_splits.append(data[bounds[i] : bounds[i + 1]])
    return data_splits


def collate(batch):
    x, y = zip(*batch)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)
