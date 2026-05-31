"""Zero-ML keyword/heuristic router over prompt text + has_image flag."""
import re

REASONING_KW = re.compile(
    r"\b(prove|proof|derive|reason|reasoning|step by step|carefully|complex|"
    r"theorem|algorithm|optimi[sz]e|eigen|integral|induction|complexity|"
    r"rigorous|justify|trade-?offs?|implications)\b", re.I)

TRY_AGAIN_KW = re.compile(
    r"\b(wrong|incorrect|not right|try again|redo|mistake|inaccurate|"
    r"incomplete|reconsider|do it again|still not)\b", re.I)

# image_question = about the *user* / their surroundings/appearance
SELF_KW = re.compile(
    r"\b(i|me|my|myself|mine)\b", re.I)
SELF_CONTEXT_KW = re.compile(
    r"\b(look|outfit|selfie|wearing|behind me|my room|my desk|my setup|"
    r"surroundings|environment|appearance|tired)\b", re.I)

GREETING_KW = re.compile(
    r"\b(hi|hey|hello|yo|sup|what'?s up|good morning|good night|good evening|"
    r"thanks|thank you|cheers|nice to meet|how are you|how was your|lol|haha|"
    r"have a (nice|good))\b", re.I)


def predict(text: str, has_image: bool = False) -> str:
    t = (text or "").strip()

    if TRY_AGAIN_KW.search(t):
        return "try_again"

    if has_image:
        # Distinguish "about me/my surroundings" vs general image understanding
        if SELF_CONTEXT_KW.search(t) or (SELF_KW.search(t) and "?" in t and len(t) < 60):
            return "image_question"
        return "image_understanding"

    if REASONING_KW.search(t):
        return "hard_question"

    # short + social -> chit_chat
    words = t.split()
    if GREETING_KW.search(t) and len(words) <= 8:
        return "chit_chat"
    if len(words) <= 3 and GREETING_KW.search(t):
        return "chit_chat"

    return "other"
