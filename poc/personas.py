"""Two hand-authored personas (run plan §02, Step 1) with topical overlap —
labmates sharing a plot thread — so their embeddings actually cluster near
each other instead of testing on noise (the article's #1 trap: never use
random/uncorrelated content for an ANN benchmark).
"""

PERSONAS = {
    "a1": {
        "agent_id": "a1",
        "name": "Mira Okonkwo",
        "occupation": "Postdoc, ceramics lab — materials science",
        "traits": "precise, conflict-avoidant, insomniac, generous with her time",
        "home": "apartment_3",
        "frequents": {"ceramics_lab": 0.6, "apartment_3": 0.4},
        "concerns": "a paper deadline on March 14; a kiln batch that keeps cracking",
        "seed_memories": [
            (4, "Mira Okonkwo is a 34-year-old materials science postdoc."),
            (5, "Mira works in the ceramics lab, usually 8am to 6pm."),
            (7, "Mira is working toward a paper deadline on March 14."),
            (6, "Mira's kiln batches have been cracking and she doesn't know why."),
            (6, "Teo Varga is Mira's labmate; she feels quietly competitive with him."),
        ],
    },
    "a2": {
        "agent_id": "a2",
        "name": "Teo Varga",
        "occupation": "PhD student, ceramics lab — materials science",
        "traits": "curious, a little impatient, collaborative, early riser",
        "home": "apartment_7",
        "frequents": {"ceramics_lab": 0.7, "apartment_7": 0.3},
        "concerns": "proving his ramp-rate hypothesis for the kiln failures; wants Mira's approval",
        "seed_memories": [
            (4, "Teo Varga is a 27-year-old PhD student in materials science."),
            (5, "Teo works in the ceramics lab alongside Mira, most days 8am to 6pm."),
            (6, "Teo suspects the kiln's ramp rate is too fast and is causing cracking."),
            (5, "Teo logged batch failures carefully, hoping to spot the pattern first."),
            (5, "Teo admires Mira's precision but wishes she'd loosen up a little."),
        ],
    },
}

LOCATIONS = {
    "ceramics_lab": "a shared materials-science lab with kilns, workbenches, and a whiteboard of batch logs",
    "apartment_3": "Mira's small apartment, quiet at night",
    "apartment_7": "Teo's apartment, a short walk from the lab",
}
