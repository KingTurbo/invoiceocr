FIELDS_CONFIG = [
    {"name": "**PRIMARY ANCHOR**", "mandatory": True},
    {"name": "**SECONDARY ANCHOR**", "mandatory": True},
    {"name": "invoice_number", "mandatory": True},
    {"name": "invoice_date", "mandatory": False},
    {"name": "total_amount", "mandatory": False}
]

HASH_MATCH_THRESHOLD = 2
HASH_SIMILARITY_THRESHOLD = 12

TRIAGE_AREA_RATIO = (0.0, 0.0, 1.0, 0.10)

# --- NEW CONFIGURABLE THRESHOLD ---
# The minimum number of ORB features an anchor selection must contain
# to be considered a valid, robust anchor during template learning.
# Lower this value if you frequently process documents with very simple logos.
# Increase it for higher reliability on complex documents.
MIN_FEATURES_FOR_VALID_ANCHOR = 5 # Changed from hardcoded 20 to a configurable 15```