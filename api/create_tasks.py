# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()
# TAIGA_URL = os.getenv("TAIGA_BASE_URL")

# def create_task(token, project_id, title, desc="Auto-generated"):
#     headers = {"Authorization": f"Bearer {token}"}
#     payload = {
#         "subject": title,
#         "description": desc,
#         "project": project_id,
#         "status": 1  # "New"
#     }
#     r = requests.post(f"{TAIGA_URL}/api/v1/tasks", headers=headers, json=payload)
#     print(f"→ {title} : {r.status_code}")
import requests
import os
from dotenv import load_dotenv

load_dotenv()
TAIGA_URL = os.getenv("TAIGA_BASE_URL")

def create_user_story(token, project_id, title, desc="Auto-generated"):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "subject": title,
        "description": desc,
        "project": project_id,
        "status": 1
    }
    r = requests.post(f"{TAIGA_URL}/api/v1/userstories", headers=headers, json=payload)
    print(f"📝 Story → {title} : {r.status_code}")
    return r.json()["id"] if r.ok else None

def create_task(token, project_id, title, desc="Auto-generated", parent_story=None):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "subject": title,
        "description": desc,
        "project": project_id,
        "status": 1
    }
    if parent_story:
        payload["user_story"] = parent_story
    r = requests.post(f"{TAIGA_URL}/api/v1/tasks", headers=headers, json=payload)
    print(f"  ↳ Task → {title} : {r.status_code}")
