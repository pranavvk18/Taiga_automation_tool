# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# TAIGA_URL = os.getenv("TAIGA_BASE_URL")

# def get_auth_token():
#     res = requests.post(f"{TAIGA_URL}/api/v1/auth", json={
#         "type": "normal",
#         "username": os.getenv("TAIGA_USERNAME"),
#         "password": os.getenv("TAIGA_PASSWORD")
#     })
#     res.raise_for_status()
#     return res.json()["auth_token"]

# def get_project_id(token, slug):
#     headers = {"Authorization": f"Bearer {token}"}
#     res = requests.get(f"{TAIGA_URL}/api/v1/projects/by_slug?slug={slug}", headers=headers)
#     res.raise_for_status()
#     return res.json()["id"]
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TAIGA_URL = os.getenv("TAIGA_BASE_URL")

def get_auth_token():
    res = requests.post(f"{TAIGA_URL}/api/v1/auth", json={
        "type": "normal",
        "username": os.getenv("TAIGA_USERNAME"),
        "password": os.getenv("TAIGA_PASSWORD")
    })
    res.raise_for_status()
    return res.json()["auth_token"]

def get_project_id(token, slug):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{TAIGA_URL}/api/v1/projects/by_slug?slug={slug}", headers=headers)
    res.raise_for_status()
    return res.json()["id"]

