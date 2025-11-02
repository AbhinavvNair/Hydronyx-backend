import json
from rapidfuzz import fuzz

# Load user credentials
with open("backend/users.json") as f:
    USERS = json.load(f)

def fuzzy_login(input_user: str, input_pass: str, threshold: int = 85):
    """
    Check login using fuzzy string matching.
    Returns True if both username and password are 'close enough'.
    """
    for stored_user, stored_pass in USERS.items():
        user_score = fuzz.ratio(input_user, stored_user)
        pass_score = fuzz.ratio(input_pass, stored_pass)

        if user_score >= threshold and pass_score >= threshold:
            return True
    return False