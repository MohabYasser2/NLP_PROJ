from src.features.char_features import char_identity_features
from src.features.word_features import word_position_features
from src.features.prefix_suffix import arabic_morph_features

def extract_features(sequence):
    """
    sequence: list of base characters (no diacritics)
    returns: list of feature dicts
    """
    feats = []

    for i in range(len(sequence)):
        f = {}
        f.update(char_identity_features(sequence, i))
        f.update(word_position_features(sequence, i))
        f.update(arabic_morph_features(sequence, i))
        feats.append(f)

    return feats
