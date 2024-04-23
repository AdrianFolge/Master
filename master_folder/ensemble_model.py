from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def ensemble_models(dict_with_predictions, refs, instances, list_of_context):
    ## Init eval model basert på kontekst
    content_list = []
    question = refs["spørsmål"]
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    # Define the prompt template
    prompt_template = PromptTemplate(
        template="""Task: Answer Evaluation
    You are given a question, six prediction answers, and one chunk of context. Your task is to determine which of the answers best answer the question based on the provided context. Only answer with the prediction name of the best fitting answer.
    - Question: {question}
    - Prediction 1: {prediction_1}
    - Prediction 2: {prediction_2}
    - Prediction 3: {prediction_3}
    - Prediction 4: {prediction_4}
    - Prediction 5: {prediction_5}
    - Prediction 6: {prediction_6}
    - Context 1: {context_1}
    Do any of the answers answer the question based on the provided context? [Prediction 1/Prediction 2/Prediction 3/Prediction 4/Prediction 5/Prediction 6]
    agent_scratchpad: This is the scratchpad where you can store intermediate information.""",
        input_variables=["question", "prediction_1", "prediction_2", "prediction_3", "prediction_4", "prediction_5", "prediction_6", "context_1"]
    )
    chain = prompt_template | llm

    for num in range(instances):
        answer = chain.invoke(
            {
                "question": question[num],
                "prediction_1": dict_with_predictions["Prediction 1"][num],
                "prediction_2": dict_with_predictions["Prediction 2"][num],
                "prediction_3": dict_with_predictions["Prediction 3"][num],
                "prediction_4": dict_with_predictions["Prediction 4"][num],
                "prediction_5": dict_with_predictions["Prediction 5"][num],
                "prediction_6": dict_with_predictions["Prediction 6"][num],
                "context_1": list_of_context[num]
            }
        )
        for prediction_key in dict_with_predictions.keys():
            if prediction_key in answer.content:
                ans = dict_with_predictions[prediction_key][num]
                content_list.append(ans)
                break 
    return content_list