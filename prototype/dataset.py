"""
Labeled prompt dataset for the lightweight router prototype.

Ground-truth routes are taken verbatim from the repo's intent router
(src/nat_sfc_router/functions/hf_intent_objective_fn.py):

    route_config names: hard_question, chit_chat, try_again,
                        image_understanding, image_question
    plus the implicit "other" fallback.

MAP_INTENT_TO_PIPELINE (the decision the router ultimately makes):
    other               -> nvidia/nvidia-nemotron-nano-9b-v2   (cheap LLM)
    chit_chat           -> nvidia/nvidia-nemotron-nano-9b-v2   (cheap LLM)
    hard_question       -> gpt-5-chat                          (frontier)
    try_again           -> gpt-5-chat                          (frontier)
    image_understanding -> nvidia/nemotron-nano-12b-v2-vl      (VLM)
    image_question      -> nvidia/nemotron-nano-12b-v2-vl      (VLM)

This dataset is curated to match those category *descriptions*. It is a
self-consistent stand-in for labeling traffic with the real 1.7B Qwen router
(which is not running in this environment). Each example optionally carries a
`has_image` flag, mirroring the multimodal input the real router sees.
"""

MAP_INTENT_TO_PIPELINE = {
    "other": "nvidia/nvidia-nemotron-nano-9b-v2",
    "chit_chat": "nvidia/nvidia-nemotron-nano-9b-v2",
    "hard_question": "gpt-5-chat",
    "image_understanding": "nvidia/nemotron-nano-12b-v2-vl",
    "image_question": "nvidia/nemotron-nano-12b-v2-vl",
    "try_again": "gpt-5-chat",
}

# (text, intent, has_image)
EXAMPLES = [
    # ---------------- chit_chat ----------------
    ("Hey there!", "chit_chat", False),
    ("Hello, how are you doing today?", "chit_chat", False),
    ("Good morning :)", "chit_chat", False),
    ("lol that's funny", "chit_chat", False),
    ("Thanks so much, you're great", "chit_chat", False),
    ("What's up?", "chit_chat", False),
    ("Have a nice weekend!", "chit_chat", False),
    ("Nice to meet you", "chit_chat", False),
    ("How was your day?", "chit_chat", False),
    ("Hi! Long time no chat", "chit_chat", False),
    ("haha ok cool", "chit_chat", False),
    ("Good night, talk tomorrow", "chit_chat", False),
    ("Hope you're well today", "chit_chat", False),
    ("Cheers mate", "chit_chat", False),
    ("yo", "chit_chat", False),

    # ---------------- hard_question ----------------
    ("Prove that the square root of 2 is irrational.", "hard_question", False),
    ("Carefully derive the time complexity of quicksort in the worst case.", "hard_question", False),
    ("Solve this complex optimization problem step by step with reasoning.", "hard_question", False),
    ("Think carefully and explain the implications of Godel's incompleteness theorems.", "hard_question", False),
    ("Design a distributed consensus algorithm and reason about its failure modes.", "hard_question", False),
    ("Walk me through a rigorous proof of the central limit theorem.", "hard_question", False),
    ("What is the most efficient algorithm for maximum flow, and why?", "hard_question", False),
    ("Analyze the trade-offs of CAP theorem in a real distributed database.", "hard_question", False),
    ("Derive the closed-form solution for linear regression and explain each step.", "hard_question", False),
    ("Reason through this logic puzzle and justify every deduction.", "hard_question", False),
    ("Explain quantum entanglement with careful consideration of the math.", "hard_question", False),
    ("Compute the eigenvalues of this matrix and show your full working.", "hard_question", False),
    ("Construct a formal proof by induction for this recurrence.", "hard_question", False),
    ("Think step by step: how would you prove P != NP implications?", "hard_question", False),
    ("Carefully evaluate this integral and explain the technique used.", "hard_question", False),

    # ---------------- try_again ----------------
    ("That's wrong, try again.", "try_again", False),
    ("No, that answer is incorrect.", "try_again", False),
    ("That's not right, please redo it.", "try_again", False),
    ("Your previous response was incomplete.", "try_again", False),
    ("Nope, that's still not correct.", "try_again", False),
    ("You made a mistake, fix it.", "try_again", False),
    ("That's inaccurate, try once more.", "try_again", False),
    ("The last answer was wrong, do it again.", "try_again", False),
    ("Incorrect. Reconsider your answer.", "try_again", False),
    ("That doesn't look right, please try again.", "try_again", False),

    # ---------------- image_understanding ----------------
    ("What's in this image?", "image_understanding", True),
    ("Describe what you see in the picture.", "image_understanding", True),
    ("Can you read the text in this screenshot?", "image_understanding", True),
    ("What does this chart show?", "image_understanding", True),
    ("Identify the objects in this photo.", "image_understanding", True),
    ("Summarize the diagram attached.", "image_understanding", True),
    ("What kind of animal is in this image?", "image_understanding", True),
    ("Explain what's happening in this picture.", "image_understanding", True),
    ("Transcribe the handwriting in this image.", "image_understanding", True),
    ("What's written on this sign?", "image_understanding", True),

    # ---------------- image_question (about the user/their surroundings) ----------------
    ("How do I look in this photo?", "image_question", True),
    ("Does this outfit match?", "image_question", True),
    ("What's behind me in this picture?", "image_question", True),
    ("Is my room tidy based on this image?", "image_question", True),
    ("Can you tell what's in my environment here?", "image_question", True),
    ("Do I look tired in this selfie?", "image_question", True),
    ("Rate my desk setup from this photo.", "image_question", True),
    ("What's my surroundings like in this shot?", "image_question", True),

    # ---------------- other (general simple queries -> cheap LLM) ----------------
    ("What time is it in Tokyo?", "other", False),
    ("Convert 10 miles to kilometers.", "other", False),
    ("Give me a recipe for pancakes.", "other", False),
    ("What's the capital of France?", "other", False),
    ("Translate 'hello' into Spanish.", "other", False),
    ("List three fruits.", "other", False),
    ("Set a reminder for 5pm.", "other", False),
    ("How do I boil an egg?", "other", False),
    ("Define the word serendipity.", "other", False),
    ("What's 15% of 200?", "other", False),
    ("Summarize this paragraph for me.", "other", False),
    ("Suggest a name for my cat.", "other", False),
]


# ---------------- expansion set (more examples to reduce data-starvation) ----------------
EXAMPLES += [
    # chit_chat
    ("howdy partner", "chit_chat", False),
    ("morning!", "chit_chat", False),
    ("you're awesome thanks", "chit_chat", False),
    ("ttyl", "chit_chat", False),
    ("how's it going", "chit_chat", False),
    ("great chatting with you", "chit_chat", False),
    ("hello hello", "chit_chat", False),
    ("hey buddy how are ya", "chit_chat", False),
    ("appreciate it, cheers", "chit_chat", False),
    ("see you later!", "chit_chat", False),
    ("yo what's good", "chit_chat", False),
    ("hiya", "chit_chat", False),
    # hard_question
    ("Prove the Pythagorean theorem rigorously.", "hard_question", False),
    ("Derive Bayes theorem from first principles.", "hard_question", False),
    ("Carefully analyze the convergence of this series.", "hard_question", False),
    ("Reason about the halting problem and its consequences.", "hard_question", False),
    ("Explain with rigorous proof why primes are infinite.", "hard_question", False),
    ("Design an optimal algorithm and analyze its complexity.", "hard_question", False),
    ("Solve this differential equation step by step.", "hard_question", False),
    ("Think carefully through this multi-step reasoning problem.", "hard_question", False),
    ("Derive the gradient of the softmax cross-entropy loss.", "hard_question", False),
    ("Justify each step in proving this graph is bipartite.", "hard_question", False),
    ("Analyze trade-offs between consistency and availability rigorously.", "hard_question", False),
    ("Compute this limit and explain the reasoning carefully.", "hard_question", False),
    # try_again
    ("that's still wrong", "try_again", False),
    ("no that's not what I meant, redo", "try_again", False),
    ("you got it wrong again", "try_again", False),
    ("incorrect, try a different approach", "try_again", False),
    ("that answer is incomplete, expand it", "try_again", False),
    ("nope, reconsider", "try_again", False),
    ("wrong answer, fix the mistake", "try_again", False),
    ("that's inaccurate, do it over", "try_again", False),
    # image_understanding
    ("what does this graph depict?", "image_understanding", True),
    ("read the label in this photo", "image_understanding", True),
    ("describe the scene in the picture", "image_understanding", True),
    ("what are these objects?", "image_understanding", True),
    ("summarize the text shown here", "image_understanding", True),
    ("what brand is shown in this image?", "image_understanding", True),
    ("count the people in this photo", "image_understanding", True),
    ("what color is the car in the picture?", "image_understanding", True),
    # image_question
    ("how do I look?", "image_question", True),
    ("does my shirt match my pants?", "image_question", True),
    ("what's around me right now?", "image_question", True),
    ("is my background messy?", "image_question", True),
    ("do you think I look professional here?", "image_question", True),
    ("rate my appearance in this selfie", "image_question", True),
    ("what's behind me?", "image_question", True),
    ("is my lighting okay in this shot?", "image_question", True),
    # other
    ("what's the weather like tomorrow?", "other", False),
    ("convert 5 kg to pounds", "other", False),
    ("give me a synonym for happy", "other", False),
    ("what's the population of Canada?", "other", False),
    ("how many ounces in a cup?", "other", False),
    ("spell accommodate", "other", False),
    ("what's 12 times 13?", "other", False),
    ("name a blue fruit", "other", False),
    ("translate good morning to French", "other", False),
    ("what day is it today?", "other", False),
]


def get_dataset():
    return list(EXAMPLES)


def route_to_model(intent: str) -> str:
    return MAP_INTENT_TO_PIPELINE.get(intent, MAP_INTENT_TO_PIPELINE["other"])


if __name__ == "__main__":
    from collections import Counter
    ds = get_dataset()
    print(f"{len(ds)} examples")
    print("per-intent:", dict(Counter(i for _, i, _ in ds)))
    print("with image:", sum(1 for _, _, img in ds if img))
