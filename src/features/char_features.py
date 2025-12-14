def char_identity_features(seq, i):
    """Character identity + context window (±2)"""
    features = {}

    features["char"] = seq[i]

    if i > 0:
        features["char-1"] = seq[i - 1]
    else:
        features["BOS"] = True

    if i > 1:
        features["char-2"] = seq[i - 2]

    if i < len(seq) - 1:
        features["char+1"] = seq[i + 1]
    else:
        features["EOS"] = True

    if i < len(seq) - 2:
        features["char+2"] = seq[i + 2]

    return features
