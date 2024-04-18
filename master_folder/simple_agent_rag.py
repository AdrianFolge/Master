import re
from deep_translator import GoogleTranslator
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

def simple_agent_rag(instances, file_path):
    loader = DirectoryLoader('../data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(docs, embeddings)
        databases[municipality_name] = db
    
    ## Running RAG with simple agent
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages

    llm = ChatOpenAI(temperature=0)

    list_of_answers_with_simple_agent = []
    for i in range(instances):
        question = references["spørsmål"][i]
        kommunenavn = references["kommunenavn"][i]
        db = databases[kommunenavn]
        retriever = db.as_retriever()

        tool = create_retriever_tool(
            retriever,
            "search_state_of_union",
            "Searches and returns excerpts from the 2022 State of the Union.",
        )
        tools = [tool]

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        result = agent_executor.invoke(
            {
                "input": f"{question}"
            }
        )
        list_of_answers_with_simple_agent.append(result["output"])
    return list_of_answers_with_simple_agent

def simple_agent_rag_translated(instances, file_path):
    loader = DirectoryLoader('../data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        for document in docs:
            page_content = document.page_content
            translated_content = GoogleTranslator(source='no', target='en').translate(text=page_content)
            document.page_content = translated_content
        for index, doc in enumerate(docs):
            if isinstance(doc.page_content, type(None)):
                docs[index].page_content = ""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(docs, embeddings)
        databases[municipality_name] = db
    ## Running RAG with simple agent
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages

    llm = ChatOpenAI(temperature=0)

    list_of_answers_with_simple_agent = []
    for i in range(instances):
        question = references["spørsmål"][i]
        kommunenavn = references["kommunenavn"][i]
        db = databases[kommunenavn]
        retriever = db.as_retriever()

        tool = create_retriever_tool(
            retriever,
            "search_state_of_union",
            "Searches and returns excerpts from the 2022 State of the Union.",
        )
        tools = [tool]

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        result = agent_executor.invoke(
            {
                "input": f"{question}"
            }
        )
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=result["output"])
        list_of_answers_with_simple_agent.append(translated_answer)
    return list_of_answers_with_simple_agent