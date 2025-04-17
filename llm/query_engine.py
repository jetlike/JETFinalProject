import openai

openai.api_key = "your-api-key-here"

def get_llm_response(user_input, object_description):
    prompt = f"The user asked: '{user_input}' about {object_description}. What is it?"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
