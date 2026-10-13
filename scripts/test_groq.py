
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

def stock_research(question):
    prompt = f"""
    You are a stock market research assistant.
    Answer the user's question clearly and simply.
    If real financial data is required but unavailable,
    say so instead of inventing figures.

    Question: {question}
    """
    response = llm.invoke(prompt)
    return response.content

question = "What is the difference between revenue and net profit?"
answer = stock_research(question)

print("\nStock Research Assistant")
print(answer)
