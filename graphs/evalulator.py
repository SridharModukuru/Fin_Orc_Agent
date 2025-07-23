from graphs.orchestrator import orchestrate
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it")


from typing import Annotated, List
import operator
from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage



class State(TypedDict):
    result: str
    topic: str
    feedback: str
    good_or_bad: str



class Feedback(BaseModel):
    grade: Literal["good", "bad"] = Field(
        description="Decide if the result is good or bad according to the query asked. also consider this in a chain, so someexchanges might be not related to input, act accordingly."
    )
    feedback: str = Field(
        description="If the result is bad, provide feedback on how to improve it."
    )

evaluator = llm.with_structured_output(Feedback)


config = {"configurable": {"thread_id": "user-thread-123"}}



def llm_call_generator(state: State):
    """
    Generates verbose output from orchestrator, including tool outputs.
    """
    if state.get("feedback"):
        msg = orchestrate(
            f"{state['topic']} but take into account the feedback: {state['feedback']}",
            config
        )
    else:
        msg = orchestrate(f"{state['topic']}",config)

    if isinstance(msg, dict):
        
        for k, v in msg.items():
            print(f"{k}: {v}", flush=True)
        return {"result": str(msg)}
    
    if hasattr(msg, "content"):
        print(msg.content, flush=True)
        return {"result": msg.content}
    print(msg, flush=True)
    return {"result": str(msg)}


def llm_call_evaluator(state: State):
    """
    Evaluates the result using LLM with structured output.
    """
    grade = evaluator.invoke(f"Grade the result {state['result']}")

    print("Evaluation:", grade, flush=True)
    return {"good_or_bad": grade.grade, "feedback": grade.feedback}

def route_result(state: State):
    if state["good_or_bad"] == "good":
        return "Accepted"
    elif state["good_or_bad"] == "bad":
        return "Rejected + Feedback"

from langgraph.graph import StateGraph, START, END

optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_result,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

optimizer_workflow = optimizer_builder.compile()

def final_result(query):

    state = optimizer_workflow.invoke({"topic": query})
    print(state["result"])
