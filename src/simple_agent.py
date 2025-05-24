from langchain_ibm import ChatWatsonx
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import asyncio
from utils.console_reader import ConsoleReader 
from langchain_core.runnables.graph import MermaidDrawMethod
#from supervisor_research import arxiv_search, web_search, document_reader_tool

# Load environment variables
load_dotenv()
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not all([WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID]):
    raise ValueError("Missing required environment variables")

#Define tools
@tool
def add_integers(a: int, b: int) -> str:
    """Adds two integers and returns the result."""
    sum_result = a + b
    return f"The sum of {a} and {b} is {sum_result}."

@tool
def count_characters(text: str) -> dict:
    """Counts the appearances of each character (letters, digits, symbols) in a string."""
    counts = {}
    for char in text:
        if char != " ":
            counts[char] = counts.get(char, 0) + 1
    return counts

tools = [add_integers, count_characters]

# Initialize ChatWatsonx model
model = ChatWatsonx(
    model_id="mistralai/mistral-large", 
    project_id=WATSONX_PROJECT_ID,
    params={"max_new_tokens": 200, "temperature": 0.7},
)

# Create ReAct agent
graph = create_react_agent(model, tools=tools)

with open("simple.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))


# Run agent with console input
async def main():
    reader = ConsoleReader(fallback="Cuanto es 2+2?")
    async for item in reader:
        prompt = item["prompt"]
        try:
            inputs = {"messages": [("user", prompt)]}
            for s in graph.stream(inputs, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        except Exception as e:
            reader.write("Error", str(e))
    reader.close()
    
if __name__ == "__main__":
    asyncio.run(main())