# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# def generate_tasks(feature_idea):
#     prompt = f"""
# Convert the following feature into 3 clear user stories using the format:
# As a [type of user], I want to [do something] so that [benefit].

# Feature: {feature_idea}
# Only return the stories, one per line.
# """
    
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content(prompt)
    
#     return response.text.strip().split("\n")
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_user_stories(feature_idea):
    prompt = f"""
Convert the following feature into 3 user stories:
Use format:
As a [type of user], I want to [do something] so that [benefit].

Feature: {feature_idea}
Return only the stories, one per line.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip().split("\n")
