# 💼 FinsightAI – Agentic Intelligence for Financial Insight & Optimization

**FinsightAI** is a powerful, agent-powered system that delivers financial analysis based on natural language queries. It uses an **orchestrator-based architecture** to manage parallel agents for tasks like financial summary, market analysis, and risk detection. Each agent's output is evaluated and optimized through an **Evaluator–Optimizer feedback loop**, ensuring only the most relevant insights are returned.

Built with **modular ReAct agents**, the system optionally includes a **retriever** for handling user-provided documents—making it a Retrieval-Augmented Generation (RAG) application.

---

## Checkout graphs/orchestrator.py and graphs/evaluator.py for the structure and agents/ for tools implementation.

## 🚀 Key Features

- **Agent Orchestration with LangGraph** – Enables structured and parallel agent workflows
- **Evaluator–Optimizer Architecture** – Refines and ranks multiple agent outputs for quality assurance
- **ReAct-Style Agents with Tools** – Agents use tools to look up, retrieve, and process data
- **Optional RAG Capability** – Uses embedded document retrieval for context-aware responses
- **Modular & Scalable** – Easily extend with more agents or tools for new financial tasks

---

## 🛠️ Tools & Technologies

| Tool/Library             | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| **LangGraph**            | For building orchestrated workflows between agents                      |
| **LangChain**            | To define and manage tool-using ReAct agents                            |
| **Groq API**             | Ultra-fast large language model for real-time response generation       |
| **Hugging Face Embeddings** | Converts text into vectors for retrieval tasks                        |
| **ChromaDB**             | Lightweight, open-source vector store for RAG                           |
| **Python**               | Core programming language for backend and orchestration logic           |
| **Flask / FastAPI**      | Web interface and backend interaction                                   |
| **.env**                 | Securely stores all API keys and secrets                                |

---

## 🧠 Architecture Overview

```text
User Query
   │
   ▼
[ Orchestrator (LangGraph) ]
   ├── FinAgent → Financial Statements
   ├── MarketAgent → Market Performance
   ├── RiskAgent → Risk Factors
   ▼
[ Evaluator – Optimizer ]
   └── Composes Best Output
   ▼
[ Optional Retrieval ]
   └── User-provided docs (RAG using ChromaDB + HF Embeddings)
