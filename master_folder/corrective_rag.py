from datasets import load_dataset
from langchain import hub
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
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

def corrective_rag_translated(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            keys: A dictionary where each key is a string.
        """

        keys: Dict[str, any]

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
            template="""You are an evaluator assessing the relevance of a retrieved document to a user question.
                    Here is the retrieved document:

                    {context}

                    Here is the user question: {question}

                    If the document contains keywords related to the user question, consider it relevant.
                    It doesn't need to be a strict test. The goal is to filter out inaccurate retrievals.
                    Provide a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
                    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["question", "context"],
        )
        chain = prompt | llm | JsonOutputParser()
        # Score
        filtered_docs = []
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
    list_of_contexts = []

    for i in range(instances):
        question = references["spørsmål"][i]
        question = GoogleTranslator(source='no', target='en').translate(text=question)
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
        pprint.pprint(value["keys"]["documents"])
        pprint.pprint(value["keys"]["generation"])
        context = value["keys"]["documents"][0].page_content
        answer = value["keys"]["generation"]
        translated_context = GoogleTranslator(source='en', target='no').translate(text=context)
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=answer)
        list_of_contexts.append(translated_context)
        list_of_answers.append(translated_answer)
    return list_of_answers, list_of_contexts

def corrective_rag(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            keys: A dictionary where each key is a string.
        """

        keys: Dict[str, any]

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
        template="""Du genererer spørsmål som er godt optimalisert for gjenfinning.
        Se på inndataene og prøv å resonnere om den underliggende semantiske intensjonen/betydningen.
        Her er det opprinnelige spørsmålet:
        {question}
        Gi et forbedret spørsmål uten noen introduksjon, bare svar med det oppdaterte spørsmålet:  """,
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
    list_of_contexts = []

    for i in range(instances):
        question = references["spørsmål"][i]
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
        pprint.pprint(value["keys"]["documents"])
        pprint.pprint(value["keys"]["generation"])
        context = value["keys"]["documents"][0].page_content
        answer = value["keys"]["generation"]
        list_of_contexts.append(context)
        list_of_answers.append(answer)
    return list_of_answers, list_of_contexts
