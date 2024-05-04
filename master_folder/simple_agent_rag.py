from deep_translator import GoogleTranslator
from datasets import load_dataset
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub

def simple_agent_rag(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
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

def simple_agent_rag_translated(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
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
            "Searches and returns excerpts from the 2022 State of the Union."
        )
        tools = [tool]

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        result = agent_executor.invoke(
            {
                "input": f"{question}"
            }
        )
        print(result)
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=result["output"])
        list_of_answers_with_simple_agent.append(translated_answer)
    return list_of_answers_with_simple_agent