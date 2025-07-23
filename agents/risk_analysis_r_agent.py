import os
import io
import requests
from contextlib import redirect_stdout
from dotenv import load_dotenv

from retriever.vectorstore import retriever_
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")




llm = ChatGroq(model="gemma2-9b-it", temperature=0)
websearch = TavilySearch()

@tool
def get_financial_ratios(ticker: str):
    """
    Use this to get key financial ratios for a company. It includes crucial risk metrics
    like debt-to-equity, current ratio (liquidity), and return on equity (profitability).
    """
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit=1&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {"error": "No ratio data found."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

@tool
def get_company_rating(ticker: str):
    """
    Use this to get the analyst rating and recommendation for a stock (e.g., strong buy, hold, sell).
    This is a direct measure of market sentiment and perceived risk.
    """
    url = f"https://financialmodelingprep.com/api/v3/rating/{ticker}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {"error": "No rating data found."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}
    

agent_logs = []

def capture_agent_output(callable_fn, *args, **kwargs):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            result = callable_fn(*args, **kwargs)
        output = buffer.getvalue()
        agent_logs.append(output)
        return result
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}"
        agent_logs.append(error_msg)
        return {"error": error_msg}

@tool
def result(query: str):
    """
    name : riskanalysis_r_agent
    Main function to analyze company risk using financial tools and also gather web info.
    Logs are captured in global `agent_logs` list.
    """
    # Main analysis agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "you are being used as a risk analyser for companies. You have access to tools to find a company's rating and other data. "
         "You can infer company ticker symbols from their names (e.g., 'apple' means 'AAPL'). "
         "you must use tools like websearch and retriever everytime. make sure result is appended clearly, "
         "nothing is cut(for further use), relevant websearch (no duplicate or companies with same name), no questions should be asked, nothing."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    prompt_web = ChatPromptTemplate.from_messages([
        ("system", 
         "use tavilysearch tool to gather information about the company if mentioned name, in 10-20 phrases."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [get_company_rating, get_financial_ratios]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # web_agnet
    agent_web = create_tool_calling_agent(llm, tools=[websearch], prompt=prompt_web)
    agent_exec_w = AgentExecutor(agent=agent_web, tools=[websearch], verbose=True)

    capture_agent_output(agent_exec.invoke, {'input': query})
    capture_agent_output(agent_exec_w.invoke, {'input': query})
    
    return agent_logs

