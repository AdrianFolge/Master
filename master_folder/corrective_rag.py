from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from typing import Dict, TypedDict
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
import pprint

def corrective_rag_translated(instances, file_path):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    loader = DirectoryLoader('../data', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
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

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            keys: A dictionary where each key is a string.
        """

        keys: Dict[str, any]

    ### Nodes ###


    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        kommunenavn = state_dict["kommunenavn"]
        db = databases[kommunenavn]
        retriever = db.as_retriever()
        documents = retriever.get_relevant_documents(question)
        transform_attempts = state_dict.get("transform_attempts", 0)  # Initialize transform_attempts if not present
        return {"keys": {"documents": documents, "question": question, "transform_attempts": transform_attempts}}



    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains generation
        """
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        transform_attempts = state_dict["transform_attempts"]
        


        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {
            "keys": {"documents": documents, "question": question, "generation": generation, "transform_attempts": transform_attempts}
        }


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with relevant documents
        """

        print("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        transform_attempts = state_dict["transform_attempts"]

        # LLM
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        prompt = PromptTemplate(
            template="""Du er en vurderer som vurderer relevansen til et hentet dokument for et brukerspørsmål. \n
            Her er det hentede dokumentet: \n\n {context} \n\n
            Her er brukerspørsmålet: {question} \n
            Hvis dokumentet inneholder nøkkelord relatert til brukerspørsmålet, vurder det som relevant. \n
            Det trenger ikke å være en streng test. Målet er å filtrere ut feilaktige hentinger. \n
            Gi en binær score 'ja' eller 'nei' for å indikere om dokumentet er relevant for spørsmålet. \n
            Gi den binære scoren som en JSON med en enkelt nøkkel 'score' og ingen innledning eller forklaring.""",
            input_variables=["question", "context"],
        )

        chain = prompt | llm | JsonOutputParser()

        # Score
        filtered_docs = []
        search = "No"  # Default do not opt for web search to supplement retrieval
        for d in documents:
            score = chain.invoke(
                {
                    "question": question,
                    "context": d.page_content,
                }
            )
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search = "Yes"  # Perform web search
                continue

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "transform_attempts": transform_attempts
            }
        }

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        transform_attempts = state_dict.get("transform_attempts", 1)

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""You generate questions that are well optimized for retrieval. \n
    Look at the inputs and try to reason about the underlying semantic intention / meaning. \n
    Here is the original question: \n
    ------- \n
    {question} \n
    ------- \n
    Provide an improved question without any introduction, just respond with the updated question:  """,
            input_variables=["question"],
        )

        # LLM
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        # Prompt
        chain = prompt | llm | StrOutputParser()
        better_question = chain.invoke({"question": question})

        # Update transform_attempts in state dictionary
        state_dict["transform_attempts"] = transform_attempts + 1

        return {
            "keys": {"documents": documents, "question": better_question, "transform_attempts": transform_attempts + 1}
        }




    def decide_to_generate(state):
        """
        Determines whether to generate an answer, retry retrieval with a transformed query, or stop if max attempts reached.

        Args:
            state (dict): The current state of the agent, including all keys.

        Returns:
            str: Next node to call
        """

        print("---DECIDE TO GENERATE---")
        state_dict = state["keys"]
        filtered_documents = state_dict["documents"]
        transform_attempts = state_dict["transform_attempts"]
        transform_attempts = state_dict.get("transform_attempts", 0)
        print(state_dict)

        if len(filtered_documents) == 0:
            # No relevant documents found
            if transform_attempts < 10:
                # Retry retrieval with a transformed query
                print(transform_attempts)
                print("---DECISION: RETRY RETRIEVAL WITH TRANSFORMED QUERY---")
                return "transform_query"
            else:
                # Max attempts reached, generate answer
                print("---DECISION: MAX ATTEMPTS REACHED, GENERATE ANSWER---")
                return "generate"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)             # Retrieve
    workflow.add_node("grade_documents", grade_documents)  # Grade documents
    workflow.add_node("generate", generate)             # Generate
    workflow.add_node("transform_query", transform_query)   # Transform query

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional edges based on relevance of documents
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",  # If no relevant documents found, transform query
            "generate": "generate",                # If relevant documents found, generate answer
        },
    )

    # Edge to handle retrying retrieval with transformed query
    workflow.add_edge("transform_query", "grade_documents")

    # Edge for generating answer after transforming query
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()

    list_of_answers = []

    for i in range(instances):
        question = references["spørsmål"][i]
        question = translated_query = GoogleTranslator(source='no', target='en').translate(text=question)
        kommunenavn = references["kommunenavn"][i]
        inputs = {
            "keys": {
                "question": question,
                "kommunenavn": kommunenavn,
            }
        }
        for output in app.stream(inputs):
            for key, value in output.items():
                # Node
                pprint.pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")

        # Final generation
        pprint.pprint(value["keys"]["generation"])
        answer = value["keys"]["generation"]
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=answer)
        list_of_answers.append(translated_answer)
    return list_of_answers


