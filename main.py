from taiga_auth import get_auth_token, get_project_id
from task_generator import generate_tasks
from create_tasks import create_task
import os
from dotenv import load_dotenv

load_dotenv()

slug = os.getenv("PROJECT_SLUG")
token = get_auth_token()
project_id = get_project_id(token, slug)

feature_idea = "Implement user profile editing and password reset features"
tasks = generate_tasks(feature_idea)

for task in tasks:
    if task.strip():
        create_task(token, project_id, task.strip())
