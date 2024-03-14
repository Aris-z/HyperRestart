


# import haienv
# haienv.set_env("hsgdr")
import numpy as np

def label_to_text(task, label):
    
    task_mapping = {
        'cola': {0: 'unacceptable', 1: 'acceptable'},
        'sst2': {0: 'negative', 1: 'positive'},
        'mrpc': {0: 'not equivalent', 1: 'equivalent'},
        'qqp': {0: 'not duplicate', 1: 'duplicate'},
        'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'qnli': {0: 'entailment', 1: 'not entailment'},
        'rte': {0: 'not entailment', 1: 'entailment'},
        'wnli': {0: 'not entailment', 1: 'entailment'}
        }
    if task.lower() == 'stsb':
        if label % 0.2 < 0.0999:
            return str(round(label - 0.05, 1))
        else:
            return str(round(label + 0.05, 1))
        
    elif task.lower() not in task_mapping.keys():
        raise ValueError('Task not supported')
    
    if task.lower() in task_mapping.keys():
        if label not in task_mapping[task.lower()]:
            raise ValueError('Label not supported')
        else:
            return task_mapping[task.lower()][label]

def text_to_label(task, text,mode='prediction',ground_truth=None):
    task_mapping = {
        'cola': {'unacceptable': 0, 'acceptable': 1},
        'sst2': {'negative': 0, 'positive': 1},
        'mrpc': {'not equivalent': 0, 'equivalent': 1},
        'qqp': {'not duplicate': 0, 'duplicate': 1},
        'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
        'qnli': {'entailment': 0, 'not entailment': 1},
        'rte': {'not entailment': 0, 'entailment': 1},
        'wnli': {'not entailment': 0, 'entailment': 1}
        }
    
    task_mapping_reverse = {
        'cola': {'unacceptable': 1, 'acceptable': 0},
        'sst2': {'negative': 1, 'positive': 0},
        'mrpc': {'not equivalent': 1, 'equivalent': 0},
        'qqp': {'not duplicate': 1, 'duplicate': 0},
        'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
        'qnli': {'entailment': 0, 'not entailment': 1},
        'rte': {'not entailment': 0, 'entailment': 1},
        'wnli': {'not entailment': 0, 'entailment': 1}
        }
    
    if task.lower() == 'stsb':
        try:
            label = float(text)
            if label % 0.2 < 0.0999:
                return round(label - 0.05, 1)
            else:
                return round(label + 0.05, 1)
        except ValueError:
            return -float(ground_truth)
        
    elif task.lower() not in task_mapping.keys():
        raise ValueError('Task not supported')
    
    elif task.lower() in task_mapping.keys():
        if text not in task_mapping[task.lower()]:
            return task_mapping_reverse[task.lower()][ground_truth]
        else:
            return task_mapping[task.lower()][text]
        
        



def calculate_score_macro_avg(scores):
    """
    we report unweighted macro average of all scores.
    :param scores_per_task: calculated in Gluer via huggingface glue evaluate package
    :return:
    """
    numerical_scores = list(scores.values())
    # calculate mean
    mean = np.mean(numerical_scores)

    score = mean

    return score  

if __name__ == '__main__':
    print(label_to_text('stsb', 2.53))
    print(text_to_label('stsb', '2.5'))
    print(text_to_label('cola', 'acceptable'))
    