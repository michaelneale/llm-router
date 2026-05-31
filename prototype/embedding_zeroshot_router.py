"""
Option C: Zero-shot embedding router (no training, no labels).

Recovers the LLM's "zero-shot flexibility" WITHOUT an LLM.

Idea:
  - Use a pre-trained sentence embedder (all-MiniLM-L6-v2, ~22.7M params, CPU).
  - Embed each ROUTE DESCRIPTION once (verbatim from the repo's route_config).
  - At query time: embed the prompt, route to the nearest route description by
    cosine similarity.
  - To add a brand-new route: just write a description and embed it. No labels,
    no retraining. <-- this is the thing TF-IDF/logreg cannot do.

The route descriptions below are copied verbatim from
src/nat_sfc_router/functions/hf_intent_objective_fn.py (route_config), plus an
"other" catch-all. Image-bearing prompts are handled by the same metadata rule
the intent router relies on (the router redacts images but uses text cues).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# Verbatim from repo route_config (descriptions are the LLM's only "spec" too)
ROUTE_DESCRIPTIONS = {
    "hard_question": "A question that requires deep reasoning, or complex problem "
                     "solving, or if the user asks for careful thinking or careful consideration",
    "chit_chat": "Any social chit chat, small talk, or casual conversation.",
    "try_again": "The user explicitly says the previous answer was incorrect or incomplete.",
    "image_understanding": "A question that requires understanding an image.",
    "image_question": "A question that requires the assistant to see the user, e.g. a "
                       "question about their appearance, environment, scene or surroundings.",
    "other": "A general or simple request that does not fit the other categories.",
}


class EmbeddingZeroShotRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2", route_descriptions=None,
                 image_routes=("image_understanding", "image_question")):
        self.model = SentenceTransformer(model_name)
        self.image_routes = set(image_routes)
        self.set_routes(route_descriptions or ROUTE_DESCRIPTIONS)

    def set_routes(self, route_descriptions: dict):
        """(Re)define routes from descriptions only. This is the zero-shot path:
        adding/removing a route here requires NO labeled data and NO retraining."""
        self.routes = list(route_descriptions.keys())
        descs = [route_descriptions[r] for r in self.routes]
        self.route_vecs = self.model.encode(descs, normalize_embeddings=True)

    def add_route(self, name: str, description: str):
        """Zero-shot add: provide a name + one sentence. No examples needed."""
        d = {r: desc for r, desc in zip(self.routes, self._descs())}
        d[name] = description
        self.set_routes(d)

    def _descs(self):
        # not stored separately; reconstruct is unnecessary because set_routes
        # always passes full dict. Kept for add_route convenience.
        return [ROUTE_DESCRIPTIONS.get(r, "") for r in self.routes]

    def predict(self, text: str, has_image: bool = False) -> str:
        qv = self.model.encode([text or ""], normalize_embeddings=True)[0]
        sims = self.route_vecs @ qv  # cosine (vectors are normalized)

        if has_image:
            # restrict to image-capable routes when an image is attached
            idxs = [i for i, r in enumerate(self.routes) if r in self.image_routes]
            if idxs:
                best = max(idxs, key=lambda i: sims[i])
                return self.routes[best]
        else:
            # exclude image-only routes when there's no image
            idxs = [i for i, r in enumerate(self.routes) if r not in self.image_routes]
            if idxs:
                best = max(idxs, key=lambda i: sims[i])
                return self.routes[best]

        return self.routes[int(np.argmax(sims))]

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
