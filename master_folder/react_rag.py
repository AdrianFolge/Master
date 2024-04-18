## Load packages
from deep_translator import GoogleTranslator
from langchain.agents import AgentExecutor, create_react_agent
from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
import re
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

def react_rag(instances, file_path):   
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]") 
    loader = DirectoryLoader('../data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    databases_FAISS = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(docs, embeddings)
        databases_FAISS[municipality_name] = db
    ## Prompt for ReACT few shots
    # Get the prompt to use - you can modify this!
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    prompt = PromptTemplate(template="""Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question. Answer in Norwegian

    Few-shot examples:
    Question: Hva er datoen for vedtaket av Kommunedelplan for sentrum av bystyret?
    Final Answer: Datoen for vedtaket av Kommunedelplan for sentrum av bystyret er 26.8.2021.

    Question: Hvor kan man finne felles bestemmelser i dokumentet om Kristiansund?
    Final Answer: Felles bestemmelser kan finnes på side 3 i dokumentet om Kristiansund.

    Question: Hva er hovedtemaet for avsnitt 3.6 i dokumentet?
    Final Answer: Hovedtemaet for avsnitt 3.6 i dokumentet er offentlige områder.

    Question: Hva er emnet for kapittel 4 i dokumentet fra Kristiansund?
    Final Answer: Emnet for kapittel 4 i dokumentet fra Kristiansund er bebyggelse og anlegg.
                            
    Begin!

    Question: {input}
    Thought: {agent_scratchpad} 
    """, input_variables=["tool_names", "tools", "input", "agent_scratchpad"])

    ## Running ReACT few shots
    list_of_answers_react = []
    for num in range(instances):
        answer_and_similar_docs = {}
        query = references["spørsmål"][num]
        kommunenavn = references["kommunenavn"][num]
        db = databases_FAISS[kommunenavn]
        found_docs = db.similarity_search(query)
        all_page_contents = []
        # Iterate over each document in found_docs
        for doc in found_docs:
            # Extract the content of the document
            content = doc.page_content
            # Append the content to the list
            all_page_contents.append(content)

        # Join all the extracted contents together into one big chunk of text
        big_chunk_of_text = '\n'.join(all_page_contents)
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            "search_planning_regulations",
            "Searches and returns excerpts planning regulation documents from different municipalities",
        )
        tools = [tool]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": {query}})
        print("########### HER ER ANSWER")
        print(answer)
        answer_and_similar_docs["svar"] = answer["output"]
        answer_and_similar_docs["kontekst"] = big_chunk_of_text
        list_of_answers_react.append(answer_and_similar_docs["svar"])
    return list_of_answers_react

def react_rag_translated(instances, file_path): 
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

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    prompt = PromptTemplate(template="""Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question. Answer in Norwegian

    Few-shot examples:
    Question: Hva er datoen for vedtaket av Kommunedelplan for sentrum av bystyret?
    Final Answer: Datoen for vedtaket av Kommunedelplan for sentrum av bystyret er 26.8.2021.

    Question: Hvor kan man finne felles bestemmelser i dokumentet om Kristiansund?
    Final Answer: Felles bestemmelser kan finnes på side 3 i dokumentet om Kristiansund.

    Question: Hva er hovedtemaet for avsnitt 3.6 i dokumentet?
    Final Answer: Hovedtemaet for avsnitt 3.6 i dokumentet er offentlige områder.

    Question: Hva er emnet for kapittel 4 i dokumentet fra Kristiansund?
    Final Answer: Emnet for kapittel 4 i dokumentet fra Kristiansund er bebyggelse og anlegg.
                            
    Begin!

    Question: {input}
    Thought: {agent_scratchpad} 
    """, input_variables=["tool_names", "tools", "input", "agent_scratchpad"])

    ## Running ReACT few shots
    list_of_answers_react = []
    for num in range(instances):
        answer_and_similar_docs = {}
        query = references["spørsmål"][num]
        kommunenavn = references["kommunenavn"][num]
        db = databases[kommunenavn]
        found_docs = db.similarity_search(query)
        all_page_contents = []
        # Iterate over each document in found_docs
        for doc in found_docs:
            # Extract the content of the document
            content = doc.page_content
            # Append the content to the list
            all_page_contents.append(content)

        # Join all the extracted contents together into one big chunk of text
        big_chunk_of_text = '\n'.join(all_page_contents)
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            "search_planning_regulations",
            "Searches and returns excerpts planning regulation documents from different municipalities",
        )
        tools = [tool]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": {query}})
        print("########### HER ER ANSWER")
        print(answer)
        answer_and_similar_docs["svar"] = answer["output"]
        answer_and_similar_docs["kontekst"] = big_chunk_of_text
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=answer_and_similar_docs["svar"])
        list_of_answers_react.append(translated_answer)
    return list_of_answers_react
