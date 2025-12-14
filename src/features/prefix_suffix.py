PREFIXES = {"و", "ف", "ب", "ك", "ل"}
SUFFIXES = {"ة", "ه", "ي", "ك", "ت", "ن"}

ALEF_FORMS = {"ا", "أ", "إ", "آ", "ٱ"}

def arabic_morph_features(seq, i):
    c = seq[i]
    features = {}

    features["is_prefix_char"] = (c in PREFIXES and (i == 0 or seq[i - 1] == " "))
    features["is_suffix_char"] = (c in SUFFIXES and (i == len(seq) - 1 or seq[i + 1] == " "))

    features["is_alef"] = c in ALEF_FORMS
    features["is_waw"] = c == "و"
    features["is_ya"] = c == "ي"
    features["is_ta_marbuta"] = c == "ة"
    features["is_hamza"] = c in {"ء", "أ", "ؤ", "ئ"}

    # Alef never carries a vowel
    features["cannot_have_vowel"] = features["is_alef"]

    # Shadda candidate (most letters except alef)
    features["can_have_shadda"] = not features["is_alef"]

    return features
