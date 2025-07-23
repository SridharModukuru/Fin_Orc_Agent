import operator
import os
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.constants import END, START, Send
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from retriever.vectorstore import retriever_



checkpointer = InMemorySaver()


from agents.fin_analysis_r_agent import fin_agent
from agents.market_output_r_agent import result as market_agent
from agents.risk_analysis_r_agent import result as risk_agent


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Orchestrator"

llm = ChatGroq(model = 'gemma2-9b-it')
tools = [fin_agent, market_agent, risk_agent]
llm_with_tools = llm.bind_tools(tools)



class Section(BaseModel):
    name: str = Field(description="Name for this section of analysis from {fin_analysis, market_output, risk_analysis}")
    description: str = Field(description="description of this section should be in such a way that, name or description belongs to (fin_analysis: use tool fin_analysis) if (market_output: use tool market_output) if (risk_analysis: use tool risk_analysis)")

class Sections(BaseModel):
    sections: List[Section] = Field(description="List of sections for the analysis. Must include fin_analysis, market_output, and risk_analysis.")

planner = llm.with_structured_output(Sections)



class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    """Orchestrator that generates a plan for the report."""
    print("---EXECUTING ORCHESTRATOR NODE---")
    report_sections = planner.invoke(
        [
            SystemMessage(content="You are a financial planning assistant. Generate a plan for the analysis of an organization based on the user's topic. You must create exactly three sections: one for fin_analysis, one for market_output, and one for risk_analysis."),
            HumanMessage(content=f"Here is the topic: {state['topic']}"),
        ]
    )
    print("---ORCHESTRATOR PLAN CREATED---")
    return {"sections": report_sections.sections}

from langchain_core.messages import AIMessage, ToolMessage


tool_map = {
    "fin_agent": fin_agent,
    "market_agent": market_agent,
    "risk_agent": risk_agent,
    "retriever": retriever_
}

def llm_call(state: WorkerState):
    """Worker that calls tools AND summarizes the result to write a section."""
    section_name = state['section'].name
    print(f"---EXECUTING WORKER NODE: {section_name}---")


    messages = [
        SystemMessage(
            content="You are a financial analyst. Use the provided tools to gather details based on the section description. Then, write a concise summary based on the tool's output. Include no preamble. Use markdown formatting."
        ),
        HumanMessage(
            content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
        ),
    ]


    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            tool_function = tool_map.get(tool_call['name'])
            if tool_function:

                tool_output = tool_function.invoke(tool_call['args'])
                messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                )

    final_response = llm.invoke(messages)
    print(f"---WORKER {section_name} FINISHED---")
    
    return {"completed_sections": [f"## {section_name.replace('_', ' ').title()}\n\n{final_response.content}"]}

def assign_workers(state: State):
    """Assign a worker to each section in the plan."""
    print("---ORCHESTRATOR ASSIGNING WORKERS---")
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

def synthesizer(state: State):
    """Synthesize the full report from all the completed sections."""
    print("---EXECUTING SYNTHESIZER NODE---")
    completed_report_sections = "\n\n---\n\n".join(state["completed_sections"])
    print("---SYNTHESIZER FINISHED---")
    return {"final_report": completed_report_sections}


orchestrator_worker_builder = StateGraph(State)


orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")


orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers)

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)



config = {"configurable": {"thread_id": "user-thread-123"}}



def orchestrate(input_str: str,config):
    orchestrator_worker = orchestrator_worker_builder.compile(checkpointer= checkpointer)
    return orchestrator_worker.invoke({"topic": input_str},config)
