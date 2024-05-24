from langchain.prompts import PromptTemplate

def ensemble_models(dict_with_predictions, refs, instances, list_of_context, llm):
    content_list = []
    question = refs["spørsmål"]
    prompt_template = PromptTemplate(
        template="""
            Oppgave: Svarvurdering
            Du har fått et spørsmål, åtte forutsagte svar og én del av sammenhengen. Din oppgave er å avgjøre hvilket av svarene som best besvarer spørsmålet basert på den gitte sammenhengen. Svar kun med navnet på den forutsagte svaret som passer best.

            Spørsmål: {question}
            Forutsigelse 1: {prediction_1}
            Forutsigelse 2: {prediction_2}
            Forutsigelse 3: {prediction_3}
            Forutsigelse 4: {prediction_4}
            Forutsigelse 5: {prediction_5}
            Forutsigelse 6: {prediction_6}
            Forutsigelse 7: {prediction_5}
            Forutsigelse 8: {prediction_6}
            Sammenheng 1: {context_1}
            Besvarer noen av svarene spørsmålet basert på den gitte sammenhengen? [Forutsigelse 1/Forutsigelse 2/Forutsigelse 3/Forutsigelse 4/Forutsigelse 5/Forutsigelse 6/Forutsigelse 7/Forutsigelse 8]
            agent_scratchpad: This is the scratchpad where you can store intermediate information.""",
        input_variables=["question", "prediction_1", "prediction_2", "prediction_3", "prediction_4", "prediction_5", "prediction_6", "prediction_7", "prediction_8", "context_1"]
    )
    chain = prompt_template | llm

    for num in range(instances):
        answer = chain.invoke(
            {
                "question": question[num],
                "prediction_1": dict_with_predictions["Forutsigelse 1"][num],
                "prediction_2": dict_with_predictions["Forutsigelse 2"][num],
                "prediction_3": dict_with_predictions["Forutsigelse 3"][num],
                "prediction_4": dict_with_predictions["Forutsigelse 4"][num],
                "prediction_5": dict_with_predictions["Forutsigelse 5"][num],
                "prediction_6": dict_with_predictions["Forutsigelse 6"][num],
                "prediction_7": dict_with_predictions["Forutsigelse 7"][num],
                "prediction_8": dict_with_predictions["Forutsigelse 8"][num],
                "context_1": list_of_context[num]
            }
        )
        for prediction_key in dict_with_predictions.keys():
            if prediction_key in answer.content:
                ans = dict_with_predictions[prediction_key][num]
                content_list.append(ans)
                break 
    return content_list