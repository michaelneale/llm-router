"""Zero-shot router: cosine-match a prompt embedding to route descriptions. No training."""
import numpy as np
from sentence_transformers import SentenceTransformer

# Descriptions from repo route_config + an "other" catch-all.
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
        self.descriptions = dict(route_descriptions)
        self.routes = list(route_descriptions.keys())
        descs = [route_descriptions[r] for r in self.routes]
        self.route_vecs = self.model.encode(descs, normalize_embeddings=True)

    def add_route(self, name: str, description: str):
        self.set_routes({**self.descriptions, name: description})

    def predict(self, text: str, has_image: bool = False) -> str:
        qv = self.model.encode([text or ""], normalize_embeddings=True)[0]
        sims = self.route_vecs @ qv
        if has_image:
            idxs = [i for i, r in enumerate(self.routes) if r in self.image_routes]
        else:
            idxs = [i for i, r in enumerate(self.routes) if r not in self.image_routes]
        if idxs:
            return self.routes[max(idxs, key=lambda i: sims[i])]
        return self.routes[int(np.argmax(sims))]

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
