## Load packages
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, context_precision)
from ragas import evaluate

def evaluate_model(preds, refs, instances):
    ja_nei_liste = []
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    # Define the prompt template
    prompt_template = PromptTemplate(
        template="""Task: Answer Evaluation
    You are given a reference answer and a predicted answer. Your task is to determine whether the predicted answer matches the reference answer correctly. It does not have to be an exact match, but it should be somewhat the same.
    - The reference answer is the correct answer.
    - The predicted answer is the answer generated by a model or provided by a user.
    Your response should indicate whether the predicted answer is correct or not.
    Reference answer: {reference}
    Predicted answer: {prediction}
    Is the predicted answer correct? [Yes/No]
    agent_scratchpad: This is the scratchpad where you can store intermediate information.""",
        input_variables=["prediction", "reference"]
    )
    chain = prompt_template | llm

    for num in range(instances):
        score = chain.invoke(
            {
                "reference": refs[num],
                "prediction": preds[num],
            }
        )
        ja_nei_liste.append(score.content)

    count_of_yes = 0

    # Iterate over content_list
    for content in ja_nei_liste:
        # Check if "Yes" is present in the content
        if "Yes" in content:
            # Increment count if "Yes" is found
            count_of_yes += 1

    print("Count of 'Yes':", count_of_yes)

    count_of_no = 0

    # Iterate over content_list
    for content in ja_nei_liste:
        # Check if "Yes" is present in the content
        if "No" in content:
            # Increment count if "Yes" is found
            count_of_no += 1

    print("Count of 'no':", count_of_no)
    percentage_of_yes = count_of_yes/(count_of_yes+count_of_no)
    return percentage_of_yes

def ragas_eval(eval_dataset):
    result = evaluate(
        eval_dataset,
        metrics =[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall
        ],
    )
    return result