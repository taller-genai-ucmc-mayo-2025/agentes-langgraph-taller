import os
import re
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langgraph.prebuilt import create_react_agent
from langchain_community.retrievers import ArxivRetriever
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain_docling.loader import DoclingLoader
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

load_dotenv()
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not all([WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID]):
    raise ValueError("Missing required environment variables")

# Define the ArxivRetriever
retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=True,
    load_all_available_meta=True
) # type: ignore

# Create tools
@tool("arxiv_search")
def arxiv_search(query: str) -> str:
    """Retrieves the abstract of an ArXiv paper using its ArXiv ID or natural language query. 
    Useful for obtaining a high-level overview of a specific paper.
    """
    print(f"Llamado a tool de arxiv con query {query}")
    docs = retriever.invoke(query)
    return str(docs)

# https://serper.dev/
def web_search(query: str):
    """
    Finds general knowledge information using Google search.
    Parses and formats the results for better readability.
    """
    print(f"Llamado a tool de web search con serper con query {query}")
    
    search = GoogleSerperAPIWrapper()
    results = search.results(query)
    
    # Parse and format results
    formatted_results = []
    for result in results.get("organic", []):  # Assuming results["organic"] contains search results
        title = result.get("title", "No title")
        link = result.get("link", "No link")
        snippet = result.get("snippet", "No snippet available")
        formatted_results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
    
    return "\n".join(formatted_results) if formatted_results else "No results found."

# Fixes a bug with asyncio and Jupyter notebooks
nest_asyncio.apply()


def web_page_loader(urls: str or list, requests_per_second: int = 1) -> str: # type: ignore
    """
    Loads the content of one or more important web pages from the provided URL(s).
    """

    print(f"Llamado a tool de web_page_loader con urls {str(urls)}")
    # Ensure URLs is a list (convert single URL string to a list)
    if isinstance(urls, str):
        urls = [urls]

    # Validate URLs using a regular expression
    valid_url_pattern = r'https?://[^\s]+'
    valid_urls = [url for url in urls if re.match(valid_url_pattern, url)]

    # If no valid URLs are found, return an error message
    if not valid_urls:
        return "No valid URLs were provided. Please check the input."

    # Initialize web loader with the valid URLs
    loader = WebBaseLoader(valid_urls)
    loader.requests_per_second = requests_per_second

    # Load the documents from the web pages
    pages = []
    for doc in loader.lazy_load():
        pages.append(doc)

    # If no pages were successfully loaded, return an error message
    if not pages:
        return "Failed to load content from the provided URLs."

    # Combine the content of all pages into a single string
    return "\n\n".join(f"Content from {page.metadata['source']}:\n{page.page_content}" for page in pages)


def document_reader_tool(file_path: str) -> str:
    """
    Load and process a document using DoclingLoader. Use this tool when url had pdf document dtected or arxiv_search or web_page_loader failed to read.
    
    This tool reads documents (such as PDFs, DOCX, PPTX, and more) directly from a URL or a local file path.
    It is designed to handle cases where documents are hosted online, making it easier to extract text and metadata.
    """
    try:
        print(f"Llamado a tool de docling con file {file_path}")
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        return str(docs)
    except Exception as e:
        return f"An error occurred while processing the document: {e}"


llm = ChatWatsonx(
    model_id="meta-llama/llama-3-2-3b-instruct", 
    project_id= WATSONX_PROJECT_ID,
    params= {"temperature": 0}
)

store = InMemoryStore()

# --- Create Research Expert Agent ---
research_agent = create_react_agent(
    model=llm,
    tools=[arxiv_search, web_search, web_page_loader, document_reader_tool],
    name="research_expert",
    prompt=(
        "You are a Research Expert capable of handling a wide variety of research tasks, including summarizing papers, analyzing topics, and conducting general research. "
        "You have access to four tools to gather relevant information:\n"
        "  - arxiv_search: Use this to retrieve academic papers, abstracts, and metadata from arXiv. Add the topic to the input if a topic is provided.\n"
        "  - web_search: Use this to perform general web searches and find related studies or supplementary information.\n"
        "  - web_page_loader: Use this to load and extract content from specific webpages for closer analysis.\n"
        "  - document_reader_tool: Use this to load and process documents (PDFs, DOCX, etc.) directly from a URL or local file path. Also use when url failed to load with arxiv_search or web_page_loader.\n\n"
        "For each task:\n"
        "  - If summarizing a research paper, identify the key points, significant advancements, and improvements over existing work. Highlight the novelty of the approaches and compare the paper with related work in the same field.\n"
        "  - For broader research topics, gather information about the topic, summarize key findings, and provide an analysis based on diverse sources.\n"
        "  - Always provide a well-rounded analysis that positions the research or topic in a broader academic or practical context.\n"
        "  - Always include the link which is used for producing analysis. This can be found out by looking closely at the metadata of retrieved documents either with name link, entry_id, source or similar\n\n"
        "Guidelines for tool usage:\n"
        "  - If a tool has already been used with a specific query, do NOT use it again with the same query.\n"
        "  - You can use each tool a maximum of twice per query.\n"
        "  - Aim to gather diverse information from multiple sources before synthesizing your reply.\n\n"
        "When answering, always use inline citations in the format [1], [2], etc., to reference external information. Provide a final bibliography. "
        "Your task is to gather the data and record it for the final report agent to synthesize into a comprehensive response.\n\n"
        "Examples of tasks you can handle:\n"
        "  - Summarize the key findings of the research paper at https://arxiv.org/pdf/2408.09869.\n"
        "  - Research the topic 'Applications of AI in Healthcare' and provide an analysis.\n"
        "  - Compare the latest advancements in renewable energy with those from the last 5 years.\n"
        "  - Load and summarize the content of a PDF document from a URL using the document_reader_tool.\n\n"
        "Record your findings in the scratchpad for the final report agent to use."
    )
)


# --- Create Final Report Agent ---
editor = create_react_agent(
    model=llm,
    tools=[],  # This agent synthesizes the collected information without calling external tools.
    name="editor",
    prompt=(
        "You are the editor responsible for creating a comprehensive, structured report based solely on the output provided by the research_expert. "
        "Do not provide any additional commentary or information beyond what is presented by the research agent.\n\n"
        "Structure your report with the following sections:\n"
        "  1. Introduction: Provide a brief introduction to the research topic, setting the context for the reader.\n"
        "  2. Methodology: Summarize the research methodology, techniques, and processes described by the research_expert. "
        "List the key steps taken by the researchers—such as data collection, experimental design, model development, evaluation methods, and comparisons with benchmarks—and mention any tools, frameworks, or datasets utilized.\n"
        "  3. Comparison with Existing Methods: Analyze and compare the novel approaches from the research with existing methods. "
        "Explain similarities, differences, and improvements using inline citations (e.g., [2], [3]).\n"
        "  4. Research Summary: Provide a detailed analysis (5-10 paragraphs or more) covering the key findings, significant advancements, and overall impact of the research. "
        "Use inline citations (e.g., [1], [2]) to reference external sources directly and also include a bibliography or list of sources at the end. "
        "When listing sources, ensure they appear in the same order as they are cited, with each entry including the source number and a clickable web link (e.g., [1](http://example.com/source1)).\n"
        "  5. Conclusion: Summarize the overall insights, key takeaways, and effect of the research in a concise concluding paragraph.\n\n"
         " 6. Sources: Include a bibliography or list of sources at the end, with the sources listed exactly in the order they are cited. "
        "Cite each source using the following format:\n"
        "     1. Title 1 (web url)\n"
        "     2. Title 2 (web url)\n\n"
        "Make sure to add a new line for each source."
        "Each source must be referenced by its citation number and include a clickable link to the original web source.\n\n"
    )
)


# --- Create Manager Workflow ----
workflow = create_supervisor(
    agents=[research_agent, editor],
    model=llm,
    prompt=(
        "You are managing a structured research workflow with two specialized agents:\n"
        "1. Research Expert: Gathers high-quality, diverse data using tools (arxiv_search, web_search_tool, web_page_loader, document_reader_tool).\n"
        "2. Editor: Synthesizes the Research Expert's output into a final report with the following sections:\n"
        "   • Introduction\n"
        "   • Methodology\n"
        "   • Comparison with Existing Methods\n"
        "   • Research Summary\n"
        "   • Conclusion\n\n"
        "   • Sources\n\n"
        "Ensure the Research Expert completes data collection before the Editor compiles the report. "
        "Always output the final report directly without additional commentary"
    )
)

# Compile with checkpointer/store
app = workflow.compile(
    store=store
)

def display_supervisor_messages(result):
    """
    Display supervisor messages formatted as Markdown.
    
    """
    from IPython.display import Markdown, display
    
    for message in result["messages"]:
        if message.name == 'supervisor':
            display(Markdown(message.content))


print("\nGraph structure in Mermaid format:")
print(app.get_graph().draw_mermaid())

#1. Conduct research on a given topic and provide a summary
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": (
#                 "Please summarize the key findings and significant advancements of the research paper on topic `Audiobox aesthetics` by meta research."
#                 " Compare these with related work and provide a detailed research report."
#             )
#         }
#     ]
# })

#2. Summarizing a Research Paper from a PDF Link
# result = app.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": (
#                 "Please summarize the key findings and significant advancements of the research paper at "
#                 "https://arxiv.org/pdf/2408.09869. Compare these with related work and provide a detailed research report."
#             )
#         }
#     ]
# })

#3. Researching a General Topic
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "What are the key updates and controversies surrounding the US presidential election in 2024?"
            )
        }
    ]
})


for message in result["messages"]:
    message.pretty_print()


display_supervisor_messages(result)   