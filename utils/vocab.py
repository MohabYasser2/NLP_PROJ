class CharVocab:
    def __init__(self):
        self.char2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2char = {0: "<PAD>", 1: "<UNK>"}

    def build(self, sequences):
        for seq in sequences:
            for ch in seq:
                if ch not in self.char2id:
                    idx = len(self.char2id)
                    self.char2id[ch] = idx
                    self.id2char[idx] = ch

    def encode(self, seq):
        return [self.char2id.get(ch, 1) for ch in seq]