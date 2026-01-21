# -*- coding: utf-8 -*-
"""
Task Generative Evaluation Demo

Created on Mon November 24 08:45:19 2025

@author: agha
"""

from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1

test_sentences = [{
    "predicted": "the book you gave earlier was very insightful",
    "reference": "I found the book that you send me very deep"
},
{
    "predicted": "internal investigation is directed toward the most recent scandal",
    "reference": "the recent scandal recently received a lot of investigation"
},
{
    "predicted": "the cat is on the mat",
    "reference": "there is a cat on the mat"
}
]


bert_scorer = BERTScorer(model_type='bert-base-uncased')
for sample in test_sentences:
    P, R, F1 = bert_scorer.score([sample['predicted']], [sample['reference']])
    print('Bert-Score P/R/F1:', P.item(), R.item(), F1.item())
    score = sentence_bleu(
        [sample['reference'].split()],
        sample['predicted'].split(),
        smoothing_function=smooth,
        weights=(0.25, 0.25, 0.25, 0.25))
    print('Bleu-Score:', score*100)