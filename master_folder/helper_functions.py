def create_predictions_dict(*lists):
    predictions_dict = {}
    for i, lst in enumerate(lists, start=1):
        key = f"Forutsigelse {i}"
        predictions_dict[key] = lst
    return predictions_dict

def average_RAGAS_score(ragas_dict):
    average_score = sum(ragas_dict.values()) / len(ragas_dict) 
    return average_score