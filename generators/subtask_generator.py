import google.generativeai as genai

def generate_subtasks(story):
    prompt = f"""
Break down this user story into 3 technical tasks for implementation:

{story}

Return only tasks, one per line.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip().split("\n")
