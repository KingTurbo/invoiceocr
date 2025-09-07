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