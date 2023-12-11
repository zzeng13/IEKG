import numpy as np
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_classification_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def convert_ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return list(map(str.strip, gen_text))


def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": corpus_bleu(output_lns, [refs_lns], **kwargs).score}


def rm_idx(seq, idx):
    return [i for i in seq if i != idx]


def post_process_eval(tgts, preds, pad_idx):
    # remove pad idx
    tgts = [rm_idx(tgt, pad_idx) for tgt in tgts]
    preds = [[p for p in pred] for pred in preds]
    # remove end idx
    end_indices = [len(p) for p in tgts]
    preds = [p[:idx] for idx, p in zip(end_indices, preds)]
    return tgts, preds


def check_seq(tar, pred):
    min_len = min([len(tar), len(pred)])
    if sum(np.equal(tar[:min_len], pred[:min_len])) == len(tar):
        return 1
    return 0

def get_seq_acc(tars, preds):
    size = len(tars)
    a = 0
    for i in range(size):
        tar = tars[i]
        pred = preds[i]
        a += check_seq(tar, pred)

    return np.float32(a/size)