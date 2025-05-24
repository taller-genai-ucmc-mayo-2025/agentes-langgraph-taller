import json
import os
import re
import time
import asyncio
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.retrievers import ArxivRetriever
from langchain.tools import tool
from utils.console_reader import ConsoleReader 

# --- Configuration ---
class Config:
    MAX_REVISIONS = 2
    MODEL_ID = "mistralai/mistral-large"
    MODEL_PARAMS = {"max_new_tokens": 2000, "temperature": 0.7}

# --- State Definition ---
class ResearchState(TypedDict):
    input: str
    topic: str
    notes: List[str]
    articles: str
    draft: str
    reviews: List[str]
    revision_count: int
    output: str

# --- Tools ---
retriever = ArxivRetriever(load_max_docs=2, get_full_documents=True, load_all_available_meta=True) 

@tool("arxiv_search")
def arxiv_search(query: str) -> str:
    """Retrieves the abstract of an ArXiv paper using its ArXiv ID or natural language query. 
    Useful for obtaining a high-level overview of a specific paper.
    """
    docs = retriever.invoke(query)
    return str(docs)

search = DuckDuckGoSearchResults(output_format="list")

@tool("duckduckgo_search")
def duckduckgo_search(query: str) -> str:
    """Searches the web using DuckDuckGo for high-level information and links."""
    result = search.invoke(query)
    formatted = "\n".join([f"- {item['title']}: {item['link']}\n  {item['snippet']}" for item in result])
    return formatted or "No search results found."

tools = [arxiv_search, duckduckgo_search]

# --- Prompt Templates ---
def get_prompts():
    """Returns prompt templates for preprocessing, research, writing, reviewing, and revising."""
    return {
        "preprocess": PromptTemplate(
            input_variables=["input"],
            template=(
                "You are a research assistant. Validate and refine the user query into a clear research topic. "
                "If the query is invalid (e.g., too vague or not research-oriented), return an error. "
                "ALWAYS reply in Spanish."
                "# User Query\n{input}\n\n"
                "Output in JSON format:\n"
                "```json\n"
                "{{\"topic\": \"refined topic\", \"error\": \"error message if invalid\"}}\n"
                "```"
            )
        ),
        "research": PromptTemplate(
            input_variables=["topic"],
            template=(
                "You are a Research Expert. Conduct research on the given topic and write findings in Markdown format.\n\n"
                "# Objectives\n"
                "1. Prioritize the latest research, authors, and noteworthy news.\n"
                "2. Include authors, release dates, and URLs where available.\n"
                "3. Provide extensive summaries for context.\n\n"
                "# Topic\n{topic}\n\n"
                "Use the tools 'arxiv_search' and 'duckduckgo_search' to gather information. "
                "Format output in Markdown with clear sections."
                "ALWAYS reply in Spanish."
            )
        ),
        "write": PromptTemplate(
            input_variables=["articles", "topic"],
            template=(
                "You are a Report Writer. Write a comprehensive report on the current state of the topic based on the provided research.\n\n"
                "# Research Findings\n{articles}\n\n"
                "# Objectives\n"
                "- Engaging introduction\n"
                "- Insightful body paragraphs with 2-3 references per section\n"
                "- Properly named sections/subtitles\n"
                "- Summarizing conclusion\n"
                "- Format: Markdown\n"
                "- ALWAYS reply in Spanish\n\n"
                "# Topic\n{topic}\n\n"
                "Ensure natural flow and include SEO keywords."
            )
        ),
        "review": PromptTemplate(
            input_variables=["draft", "topic"],
            template=(
                "You are a research editor reviewing a report on {topic}.\n\n"
                "Report:\n{draft}\n\n"
                "Evaluate for completeness (covers key aspects), clarity (well-structured), and quality (engaging, accurate). ALWAYS reply in Spanish."
                "Provide 2–3 specific suggestions for improvement (e.g., missing information, unclear sections, or quality enhancements). "
                "State if further revision is needed.\n\n"
                "Format your response as:\n"
                "* Suggestion 1\n"
                "* Suggestion 2\n"
                "...\n"
                "Further Revision Needed: Yes/No"
            )
        ),
        "revise": PromptTemplate(
            input_variables=["draft", "review", "topic"],
            template=(
                "Revise the report on {topic} based on the feedback provided. Address ALL suggestions. ALWAYS reply in Spanish.\n\n"
                "Original Report:\n{draft}\n\n"
                "Feedback:\n{review}\n\n"
                "Revised Report:"
            )
        )
    }

# --- LLM Initialization ---
def initialize_llm():
    """Initializes the Watsonx LLM."""
    load_dotenv()
    env_vars = {
        "WATSONX_APIKEY": os.getenv("WATSONX_APIKEY"),
        "WATSONX_URL": os.getenv("WATSONX_URL"),
        "WATSONX_PROJECT_ID": os.getenv("WATSONX_PROJECT_ID")
    }
    if not all(env_vars.values()):
        raise ValueError("Missing Watsonx environment variables")
    return ChatWatsonx(
        model_id=Config.MODEL_ID,
        project_id=env_vars["WATSONX_PROJECT_ID"],
        params=Config.MODEL_PARAMS
    )

# --- Graph Nodes ---
def preprocess_topic(state: ResearchState, llm, prompts) -> ResearchState:
    """Validates and refines the user topic."""
    print(f"\n📝 Preprocessing topic: {state['input']}")
    chain = prompts["preprocess"] | llm
    response = chain.invoke({
        "input": state["input"]
    })
    # print("===================================== preprocess response")
    # print(response)
    # print("===================================== preprocess response")
    try:
        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.content.strip())
        # print("===================================== preprocess cleaned_response")
        # print(cleaned_response)
        # print("===================================== preprocess cleaned_response")
        result = json.loads(cleaned_response)
        if "error" != "" in result:
            return {**state, "output": result["error"]}
        return {
            **state,
            "topic": result.get("topic", state.get("topic", ""))
        }
    except json.JSONDecodeError:
        return {**state, "output": "Invalid topic provided."}

def research(state: ResearchState, llm, prompts) -> ResearchState:
    """Conducts research using a ReAct agent."""
    if not state.get("topic"):
        return {**state, "output": "No valid topic provided."}
    print(f"\n🔍 Researching topic: {state['topic']}")
    research_agent = create_react_agent(model=llm, tools=tools)
    prompt = prompts["research"].format(topic=state["topic"])
    inputs = {"messages": [("user", prompt)]}
    for s in research_agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if not isinstance(message, tuple):
            state["articles"] = message.content
    print("\nResearch Findings:")
    print(state["articles"])
    print("-" * 50)
    time.sleep(0.5)
    return state

def write_report(state: ResearchState, llm, prompts) -> ResearchState:
    """Writes a comprehensive report."""
    if not state.get("articles"):
        return {**state, "output": "No research findings provided."}
    print(f"\n✍️ Writing report for: {state['topic']}")
    chain = prompts["write"] | llm
    response = chain.invoke({
        "articles": state["articles"],
        "topic": state["topic"]
    })
    state["draft"] = response.content
    state["output"] = response.content
    print("\nReport Draft:")
    print(state["draft"])
    print("-" * 50)
    time.sleep(0.5)
    return state

def review_report(state: ResearchState, llm, prompts) -> ResearchState:
    """Reviews the report for quality."""
    print(f"\n🔎 Reviewing report (Revision {state['revision_count']})")
    chain = prompts["review"] | llm
    review = chain.invoke({"draft": state["draft"], "topic": state["topic"]})
    state["reviews"].append(review.content)
    print("\nReview:")
    print(review.content)
    print("-" * 50)
    time.sleep(0.5)
    return state

def revise_report(state: ResearchState, llm, prompts) -> ResearchState:
    """Revises the report based on the latest review."""
    print(f"\n✍️ Revising report (Revision {state['revision_count'] + 1})")
    chain = prompts["revise"] | llm
    revised = chain.invoke({
        "draft": state["draft"],
        "review": state["reviews"][-1],
        "topic": state["topic"]
    })
    state["revision_count"] += 1
    state["draft"] = revised.content
    state["output"] = revised.content
    print("\nRevised Report:")
    print(revised.content)
    print("-" * 50)
    time.sleep(0.5)
    return state

# --- Graph Definition ---
def build_graph(llm, prompts) -> StateGraph:
    """Builds the research workflow graph."""
    workflow = StateGraph(ResearchState)
    workflow.add_node("preprocess", lambda state: preprocess_topic(state, llm, prompts))
    workflow.add_node("research", lambda state: research(state, llm, prompts))
    workflow.add_node("write", lambda state: write_report(state, llm, prompts))
    workflow.add_node("review", lambda state: review_report(state, llm, prompts))
    workflow.add_node("revise", lambda state: revise_report(state, llm, prompts))

    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "review")

    def should_continue(state: ResearchState) -> str:
        """Determines whether to revise or end."""
        if state.get("output") and not state.get("draft"):
            print("Error occurred. Stopping.")
            return END
        if state["revision_count"] >= Config.MAX_REVISIONS:
            print(f"Max revisions ({Config.MAX_REVISIONS}) reached. Final report complete.")
            return END
        if not state["reviews"]:
            print("No reviews found. Stopping.")
            return END
        if "further revision needed: no" in state["reviews"][-1].lower():
            print("No further revision needed. Final report complete.")
            return END
        print("Revision needed. Continuing.")
        return "revise"

    workflow.add_conditional_edges("review", should_continue, {"revise": "revise", END: END})
    workflow.add_edge("revise", "review")
    return workflow

# --- Main Execution ---
async def main():
    try:
        llm = initialize_llm()
        prompts = get_prompts()
        graph = build_graph(llm, prompts).compile()

        
        print("\nGraph structure in Mermaid format:")
        print(graph.get_graph().draw_mermaid())

        reader = ConsoleReader()
        reader.write(
            "ℹ️",
            "I am a research and report agent. Please give me a topic for which I will write a report on its current state."
        )
        last_result = {}
        state_progress = []

        async for item in reader:
            prompt = item["prompt"]
            state = {
                "input": prompt,
                "topic": last_result.get("topic", ""),
                "notes": last_result.get("notes", []),
                "articles": "",
                "draft": "",
                "reviews": [],
                "revision_count": 0,
                "output": ""
            }
            result = graph.invoke(state)
            last_result = result
            reader.write("🤖 Answer", result["output"])
            state_progress.append({"step": "Final Answer", "state": result})

            # Save progress to Markdown
            markdown_output = ""
            for i, progress in enumerate(state_progress):
                markdown_output += f"\n## Step {i+1}: {progress['step']}\n\n"
                for key, value in progress["state"].items():
                    if value and key != "reviews" or (key == "reviews" and value):
                        markdown_output += f"### {key.capitalize()}:\n{value}\n"
                markdown_output += "\n---\n"
            topic_safe = "".join(c for c in result["topic"] if c.isalnum() or c in (" ", "_")).replace(" ", "_")
            with open(f"state_progress_{topic_safe or 'report'}.md", "w", encoding="utf-8") as f:
                f.write(markdown_output)
            reader.write("", f"State progress saved to state_progress_{topic_safe or 'report'}.md")
        reader.close()

    except Exception as e:
        reader.write("Error", str(e))

if __name__ == "__main__":
    asyncio.run(main())