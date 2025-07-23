from retriever.vectorstore import retriever_ 
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
import os
import io
import sys
from contextlib import redirect_stdout
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

websearch = TavilySearch()  
llm = ChatGroq(model="gemma2-9b-it", temperature=0)


@tool
def get_company_profile(ticker: str):
    '''
    name: get_company_profile
    this function must be used this function should only be used if proper and surely ticker is available for the company
    gets general comany profile details
    input : ticker :str
    returns api response as text'''
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]
        return {"error": "No data found for this ticker."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}


@tool
def get_annual_financial_statements(ticker: str):
    '''
    name:get_annual_financial_statements
    this function should only be used if proper and surely ticker is available for the company
    gets annual financial statements of company 
    input : ticker :str
    return api response as text'''
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&limit=5&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]
        return {"error": "No financial statements found for this ticker."}
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
def fin_agent(query):
    '''
    name:fin_analysis_r_agent
    input: str query
    output: financial details of the company'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a help ful finance assistant, if you are sure then use ticker name of the company to query, else just return empty, donot give false information taking the wrong ticker name."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    prompt_web = ChatPromptTemplate.from_messages([
        ("system", "you are a help ful finance assistant use tavilysearch tool for getting company details donot use any other tools"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    tools = [
        # tool(retriever_),
        get_annual_financial_statements,
        get_company_profile
    ]
    tools_web = [websearch]

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=True)

    web_Agent = create_tool_calling_agent(llm, tools_web, prompt_web)
    agent_exec_web = AgentExecutor(agent=web_Agent, tools=tools_web, verbose=True)

    response = capture_agent_output(agent_exec.invoke, {'input': query})
    return agent_logs