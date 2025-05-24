from langgraph.graph import StateGraph, END
from typing import TypedDict
import random
from langchain_core.runnables.graph import MermaidDrawMethod

# Define state schema
class WorkflowState(TypedDict):
    hops: int
    next_step: str

# Workflow steps
def step_a(state: WorkflowState) -> WorkflowState:
    print(f"-> start step_a")
    state["hops"] += 1
    return state

def step_b(state: WorkflowState) -> WorkflowState:
    print(f"-> start step_b")
    value = random.random()
    print(f"Valor aleatorio {value}")
    if value < 0.5:
        state["next_step"] = "step_a"
    else:
        state["next_step"] = END
    return state

def should_continue(state: WorkflowState) -> str:
    next_step = state["next_step"]
    if next_step == "step_a":
        return "step_a"
    else:
        return END


# Define the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("a", step_a)
workflow.add_node("b", step_b)
workflow.add_edge("a", "b")
workflow.add_conditional_edges("b", should_continue, {"step_a": "a", END: END})
workflow.set_entry_point("a")

# Compile the graph
graph = workflow.compile()

with open("workflow_simple.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))


# Run the workflow
state = {"hops": 0}
result = graph.invoke(state)

# Log results
print(f"Hops: {result['hops']}")
