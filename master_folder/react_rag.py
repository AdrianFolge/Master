## Load packages
from deep_translator import GoogleTranslator
from langchain.agents import AgentExecutor, create_react_agent
from datasets import load_dataset
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate

def react_rag(instances, file_path, databases, llm):   
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]") 
    prompt = PromptTemplate(template="""Svar på det følgende spørsmålet så godt du kan. Du kan bruke følgende verktøy:

    {tools}

    Bruk følgende format:

    Spørsmål: Spørsmålet du må besvare
    Tanke: Du burde alltid tenke på det du gjør
    Handling: handlingen å utføre, skal være en av [{tool_names}]
    Handling Inndata: inndata til handlingen
    Observasjon: resultatet av handlingen
    ... (denne Tanke/Handling/Handling Inndata/Observasjon kan gjenta seg N ganger)
    Tanke: Jeg vet nå det endelige svaret
    Endelig Svar: det endelige svaret på det opprinnelige inndata spørsmålet. Svar på norsk.

    Few-shot eksempler:
    Spørsmål: Hva er datoen for vedtaket av Kommunedelplan for sentrum av bystyret?
    Endelig Svar: Datoen for vedtaket av Kommunedelplan for sentrum av bystyret er 26.8.2021.

    Spørsmål: Hvor kan man finne felles bestemmelser i dokumentet om Kristiansund?
    Endelig Svar: Felles bestemmelser kan finnes på side 3 i dokumentet om Kristiansund.

    Spørsmål: Hva er hovedtemaet for avsnitt 3.6 i dokumentet?
    Endelig Svar: Hovedtemaet for avsnitt 3.6 i dokumentet er offentlige områder.

    Spørsmål: Hva er emnet for kapittel 4 i dokumentet fra Kristiansund?
    Endelig Svar: Emnet for kapittel 4 i dokumentet fra Kristiansund er bebyggelse og anlegg.
                            
    Begynn!

    Spørsmål: {input}
    Tanke: {agent_scratchpad} 
    """, input_variables=["tool_names", "tools", "input", "agent_scratchpad"])

    ## Running ReACT few shots
    list_of_context = []
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



        answer_and_similar_docs["svar"] = answer["output"]
        answer_and_similar_docs["kontekst"] = big_chunk_of_text
        list_of_answers_react.append(answer_and_similar_docs["svar"])
        list_of_context.append(answer_and_similar_docs["kontekst"])
    return list_of_answers_react, list_of_context

def react_rag_translated(instances, file_path, databases, llm): 
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
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

    Question: What is the date of the decision on the Municipal Plan for the city center by the city council?
    Final Answer: The date of the decision on the Municipal Plan for the city center by the city council is 26.8.2021.

    Question: Where can one find common provisions in the document about Kristiansund?
    Final Answer: Common provisions can be found on page 3 in the document about Kristiansund.

    Question: What is the main topic of section 3.6 in the document?
    Final Answer: The main topic of section 3.6 in the document is public areas.

    Question: What is the subject of chapter 4 in the document from Kristiansund?
    Final Answer: The subject of chapter 4 in the document from Kristiansund is buildings and facilities.
                            
    Begin!

    Question: {input}
    Thought: {agent_scratchpad} 
    """, input_variables=["tool_names", "tools", "input", "agent_scratchpad"])

    ## Running ReACT few shots
    list_of_answers_react = []
    list_of_contexts_react = []
    for num in range(instances):
        answer_and_similar_docs = {}
        query = references["spørsmål"][num]
        kommunenavn = references["kommunenavn"][num]
        db = databases[kommunenavn]
        translated_query = GoogleTranslator(source='no', target='en').translate(text=query)
        found_docs = db.similarity_search(translated_query)
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
        answer = agent_executor.invoke({"input": {translated_query}})

       
        answer_and_similar_docs["svar"] = answer["output"]
        answer_and_similar_docs["kontekst"] = big_chunk_of_text
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=answer_and_similar_docs["svar"])
        translated_context = GoogleTranslator(source='en', target='no').translate(text=answer_and_similar_docs["kontekst"])
        list_of_answers_react.append(translated_answer)
        list_of_contexts_react.append(translated_context)
    return list_of_answers_react, list_of_contexts_react
