def word_position_features(seq, i):
    features = {}

    # Beginning / end of word
    features["is_word_start"] = (i == 0 or seq[i - 1] == " ")
    features["is_word_end"] = (i == len(seq) - 1 or seq[i + 1] == " ")

    # Position bucket
    if features["is_word_start"]:
        features["pos"] = "START"
    elif features["is_word_end"]:
        features["pos"] = "END"
    else:
        features["pos"] = "MID"

    return features
