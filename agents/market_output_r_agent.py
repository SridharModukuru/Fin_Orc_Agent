from retriever.vectorstore import retriever_
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
import os
import io
from contextlib import redirect_stdout
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv



load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")




llm = ChatGroq(model="gemma2-9b-it", temperature=0)
websearch = TavilySearch()

@tool
def get_market_movers():
    '''
    name: get_market_movers
    this function gets top gainers and losers in the market
    no input required
    returns a dictionary of top 5 gainers and losers
    '''
    url_gainers = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={FMP_API_KEY}"
    url_losers = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={FMP_API_KEY}"
    try:
        gainers = requests.get(url_gainers)
        losers = requests.get(url_losers)
        gainers.raise_for_status()
        losers.raise_for_status()
        return {
            "top_gainers": gainers.json()[:5],
            "top_losers": losers.json()[:5]
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

@tool
def get_stock_news(ticker: str, limit: int = 5):
    '''
    name: get_stock_news
    this tool gets recent news for a stock
    input: ticker (str), limit (int)
    return: list of news articles
    '''
    url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
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
    '''
    name: market_analysis_r_agent
    main function for querying stock market data and company news
    input: query (str)
    output: captured logs and agent response
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "you are being used as an info gatherer, You have access to tools to find a company's market-output data. "
         "You can infer company ticker symbols from their names (e.g., 'apple' means 'AAPL'). "
         "you must use tools like websearch and retriever everytime. make sure result is appended clearly,"
         "nothing is cut,avoid duplicate companies or with same name in webserach, no questions should be asked,nothing."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    prompt_web = ChatPromptTemplate.from_messages([
        ("system", "use tavilysearch tool to gather information about the company if mentioned name, in 10-20 phrases"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [get_market_movers, get_stock_news]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=True)
    

    agent_web = create_tool_calling_agent(llm, tools=[websearch], prompt=prompt_web)
    agent_exec_w = AgentExecutor(agent=agent_web, tools=[websearch], verbose=True)

    response = capture_agent_output(agent_exec.invoke, {'input': query})
    response_web = capture_agent_output(agent_exec_w.invoke, {'input': query})
    return agent_logs


# print(result('what is for google.'))