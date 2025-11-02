from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/lg_agent/.env", override=True)

client  = OpenAI()

response = client.responses.create(
    model="gpt-5-nano-2025-08-07",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)