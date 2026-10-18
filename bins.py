"""Segregation rules: maps the model's 5 detected classes to a suggested bin.

These rules are NOT part of the trained model. Editing this file changes only
the advice shown in the app, never what the model detects. Adjust the text to
match the recycling rules of your own city or campus.
"""

BIN_RULES = {
    "Glass": {
        "bin": "Glass recycling",
        "color": "#2e7d32",
        "note": "Accepted only where a glass stream exists. Broken glass, mirrors and ceramics usually are not.",
    },
    "Metal": {
        "bin": "Metal recycling",
        "color": "#546e7a",
        "note": "Rinse cans and tins. Aerosols, paint tins and mixed items may need special handling.",
    },
    "Paper": {
        "bin": "Paper recycling",
        "color": "#1565c0",
        "note": "Only if clean and dry. Greasy or wet paper and tissues go to general waste.",
    },
    "Plastic": {
        "bin": "Plastic recycling",
        "color": "#ef6c00",
        "note": "Depends on the plastic type and what your local service accepts. Check the recycling code.",
    },
    "Waste": {
        "bin": "General waste / manual review",
        "color": "#6d4c41",
        "note": "Not recyclable by this system. A person should check items the model cannot place.",
    },
}

DEFAULT_RULE = {
    "bin": "Manual review",
    "color": "#757575",
    "note": "No rule defined for this class.",
}


def get_bin(class_name):
    """Return the rule dict for a class name, or a safe default."""
    return BIN_RULES.get(class_name, DEFAULT_RULE)
