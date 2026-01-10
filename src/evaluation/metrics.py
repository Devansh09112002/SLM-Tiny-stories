from bert_score import score as bert_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import CHRF

def calculate_metrics(predictions, references):
    """
    Calculates a suite of standard NLP metrics.
    Args:
        predictions (list of str): The generated texts.
        references (list of str): The ground truth texts.
    Returns:
        dict: A dictionary of scores.
    """
    metrics = {}

    # BERTScore
    
    p, r, f1 = bert_scorer(predictions, references, lang="en", verbose=False, rescale_with_baseline=True)
    metrics['bert_score_f1'] = f1.mean().item()

    # ROUGE
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0
    for pred, ref in zip(predictions, references):
        scores = rouge.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
    metrics['rouge1'] = rouge1 / len(predictions)
    metrics['rouge2'] = rouge2 / len(predictions)
    metrics['rougeL'] = rougeL / len(predictions)
    
    # BLEU
    
    bleu_score = 0
    chencherry = SmoothingFunction()
    for pred, ref in zip(predictions, references):
        bleu_score += sentence_bleu([ref.split()], pred.split(), smoothing_function=chencherry.method1)
    metrics['bleu'] = bleu_score / len(predictions)
    
    # CHRF++
    
    chrf = CHRF(word_order=2) # CHRF++
    metrics['chrf++'] = chrf.corpus_score(predictions, [references]).score

    return metrics