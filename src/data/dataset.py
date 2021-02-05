import pprint
from abc import ABC, abstractmethod

import deepsmiles
import selfies as sf
import torch
from torch.utils import data


class Vocab(ABC):

    def __init__(self,
                 alphabet,
                 bos_token, eos_token, pad_token,
                 tokenize_fn):
        alphabet = list(sorted(alphabet))
        alphabet.insert(0, pad_token)
        alphabet.insert(1, bos_token)
        alphabet.insert(2, eos_token)

        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.bos_idx = self.stoi[bos_token]
        self.eos_idx = self.stoi[eos_token]
        self.pad_idx = self.stoi[pad_token]

        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.stoi)

    def __str__(self):
        return pprint.pformat(self.itos)

    def label_encode(self, s):
        x = [self.stoi[c] for c in self.tokenize_fn(s)]
        x.insert(0, self.bos_idx)
        x.append(self.eos_idx)
        return torch.tensor(x, dtype=torch.long)

    def label_decode(self, x: torch.Tensor):
        symbols = []
        for i in x:
            if i == self.eos_idx:
                break
            symbols.append(self.itos[i.item()])

        s = ''.join(symbols)
        s = s.replace(self.bos_token, "")
        s = s.replace(self.pad_token, "")
        return s

    def label_decode_batch(self, x: torch.Tensor):
        assert len(x.size()) == 2
        return [self.label_decode(x_i) for x_i in x]

    @abstractmethod
    def translate_to_smiles(self, s):
        raise NotImplementedError()


class SMILESVocab(Vocab):

    def __init__(self, lines):
        alphabet = set()
        for s in lines:
            alphabet.update(set(s))

        super().__init__(
            alphabet=alphabet,
            bos_token='[BOS]',
            eos_token='[EOS]',
            pad_token='[PAD]',
            tokenize_fn=list
        )

    def translate_to_smiles(self, s):
        return s


class DeepSMILESVocab(Vocab):

    def __init__(self, lines):
        alphabet = set()
        for s in lines:
            alphabet.update(set(s))

        super().__init__(
            alphabet=alphabet,
            bos_token='[BOS]',
            eos_token='[EOS]',
            pad_token='[PAD]',
            tokenize_fn=list
        )

        self.converter = deepsmiles.Converter(rings=True, branches=True)

    def translate_to_smiles(self, s):
        try:
            return self.converter.decode(s)
        except deepsmiles.DecodeError:
            return None
        except Exception:
            print(s)
            return None


class SELFIESVocab(Vocab):

    def __init__(self, lines):
        alphabet = sf.get_alphabet_from_selfies(lines)

        super().__init__(
            alphabet=alphabet,
            bos_token='[bos]',
            eos_token='[eos]',
            pad_token='[nop]',
            tokenize_fn=(lambda x: list(sf.split_selfies(x)))
        )

    def translate_to_smiles(self, s):
        return sf.decoder(s)


class LineByLineDataset(data.Dataset):

    def __init__(self, lines, vocab):
        self.vocab = vocab
        self.dataset = list(map(self.vocab.label_encode, lines))
        self.max_len = max(x.size(0) for x in self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
