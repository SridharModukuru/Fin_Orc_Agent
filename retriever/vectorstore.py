from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool


import os

@tool
def retriever_(input):
    '''
    This function must only be used if input doesnot contain any company details 
    This function retrieves data from the user's input file.
    The context and contents of the file are unknown: it might be a cash flow statement, company funds, or their plans.
    The file type is checked among .txt, .pdf, or .csv.
    '''

    loader = None
    if os.path.exists('file.txt'):
        loader = TextLoader('file.txt')
    elif os.path.exists('file.pdf'):
        loader = PyPDFLoader('file.pdf')
    elif os.path.exists('file.csv'):
        loader = CSVLoader('file.csv')
    else:
        raise FileNotFoundError("No input file found with supported extension.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(chunks, embedding=embeddings)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 2}
    )
    results = retriever.invoke(input)
    return [doc.page_content for doc in results]


# print(retriever_('tell me about the company'))
