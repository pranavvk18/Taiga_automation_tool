import google.generativeai as genai

def predict_tags_priority(story):
    prompt = f"""
For the user story below:
"{story}"

Predict:
1. Priority: High / Medium / Low
2. Tags: up to 3 keywords like frontend, backend, auth

Return in this format:
Priority: <priority>
Tags: <tag1>, <tag2>, ...
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()
