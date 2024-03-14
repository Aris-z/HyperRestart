# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE benchmark metric. """
import datasets
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from rouge_score import rouge_scorer
import evaluate


_CITATION = """\
@inproceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
"""

_DESCRIPTION = """\
GLUE, the General Language Understanding Evaluation benchmark
(https://gluebenchmark.com/) is a collection of resources for training,
evaluating, and analyzing natural language understanding systems.
"""

_KWARGS_DESCRIPTION = """
Compute GLUE evaluation metric associated to each GLUE dataset.
Args:
    predictions: list of predictions to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
    "accuracy": Accuracy
    "f1": F1 score
    "pearson": Pearson Correlation
    "spearmanr": Spearman Correlation
    "matthews_correlation": Matthew Correlation
Examples:
    >>> glue_metric = evaluate.load('glue', 'sst2')  # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}
    >>> glue_metric = evaluate.load('glue', 'mrpc')  # 'mrpc' or 'qqp'
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0, 'f1': 1.0}
    >>> glue_metric = evaluate.load('glue', 'stsb')
    >>> references = [0., 1., 2., 3., 4., 5.]
    >>> predictions = [0., 1., 2., 3., 4., 5.]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print({"pearson": round(results["pearson"], 2), "spearmanr": round(results["spearmanr"], 2)})
    {'pearson': 1.0, 'spearmanr': 1.0}
    >>> glue_metric = evaluate.load('glue', 'cola')
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'matthews_correlation': 1.0}
"""
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

# One prediction to multi-reference:
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def metric_groudtruth(metric_fn, prediction, ground_truth):
    return  metric_fn(prediction, ground_truth)


def rouge(prediction, ground_truth):
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(
        references
    ), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold
        )
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Glue():
    def __init__(self,task_name):
        self.task_name = task_name
        if self.task_name not in [
            'glue/cola:2.0.0', 
            'glue/sst2:2.0.0', 
            'glue/mrpc:2.0.0', 
            'glue/qqp:2.0.0', 
            'glue/stsb:2.0.0', 
            'glue/mnli:2.0.0', 
            'glue/qnli:2.0.0', 
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["glue/cola:2.0.0", "glue/sst2:2.0.0", "glue/mrpc:2.0.0", "glue/qqp:2.0.0", '
                '"glue/stsb:2.0.0", "glue/mnli:2.0.0", "glue/qnli:2.0.0"]'
            )

    def compute(self, predictions, references):
        if self.task_name in [
            'glue/cola:2.0.0', 
            'glue/sst2:2.0.0', 
            'glue/mrpc:2.0.0', 
            'glue/qqp:2.0.0', 
            'glue/stsb:2.0.0', 
            'glue/mnli:2.0.0', 
            'glue/qnli:2.0.0', 
        ]:
            assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
            
            rougeL = 0
            for pred, gold in zip(predictions, references):
                # assert isinstance(gold, list)
                rougeL += metric_groudtruth(
                    rouge, prediction=pred, ground_truth=gold
                )
            rougeL = 100.0 * rougeL / len(references)
            return {'RougeL': round(rougeL, 5)}
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["glue/cola:2.0.0", "glue/sst2:2.0.0", "glue/mrpc:2.0.0", "glue/qqp:2.0.0", '
                '"glue/stsb:2.0.0", "glue/mnli:2.0.0", "glue/qnli:2.0.0"]'
            )


class RougeMetric():
    def __init__(self):
        pass

    def compute(self, predictions, references):
        assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
        
        rougeL = 0
        for pred, gold in zip(predictions, references):
            # assert isinstance(gold, list)
            rougeL += metric_groudtruth(
                rouge, prediction=pred, ground_truth=gold
            )
        rougeL = 100.0 * rougeL / len(references)
        return {'RougeL': round(rougeL, 5)}



