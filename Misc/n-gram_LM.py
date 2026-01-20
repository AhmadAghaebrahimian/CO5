# -*- coding: utf-8 -*-
"""
Task N-gram LM Demo

Created on Thu November 27 15:27:42 2025

@author: agha
"""

import math
import nltk
import random
from nltk.corpus import brown
from collections import Counter, defaultdict


random.seed(42)


def train_bigram_model(sentences):
    counts = defaultdict(Counter)
    vocab = set()
    for sent in sentences:
        sent = ["<s>"] + sent + ["</s>"]
        for w1, w2 in zip(sent[:-1], sent[1:]):
            counts[w1][w2] += 1
            vocab.add(w2)
    return counts, vocab

def bigram_prob_add_one(counts, vocab, w1, w2, k=1.0):
    total = sum(counts[w1].values()) + k * len(vocab)
    return (counts[w1][w2] + k) / total


def generate_sentence(counts, vocab, max_len=20):
    w1 = "<s>"
    result = []
    for _ in range(max_len):
        next_words = list(counts[w1].keys()) or list(vocab)
        probs = [bigram_prob_add_one(counts, vocab, w1, w2) for w2 in next_words]
        total = sum(probs)
        probs = [p/total for p in probs]
        w1 = random.choices(next_words, weights=probs)[0]
        if w1 == "</s>":
            break
        result.append(w1)
    return " ".join(result)


def perplexity(counts, vocab, test_sents):
    logprob, N = 0, 0
    for sent in test_sents:
        sent = ["<s>"] + sent + ["</s>"]
        for w1, w2 in zip(sent[:-1], sent[1:]):
            logprob += math.log(bigram_prob_add_one(counts, vocab, w1, w2))
            N += 1
    return math.exp(-logprob / N)


if __name__ == "__main__":
    corpus = [
        ["the", "cat", "sat"],
        ["the", "cat", "slept"],
        ["the", "dog", "barked"]
    ]

    counts, vocab = train_bigram_model(corpus)
    print('Example Corpus ====================')
    print("Generated:", generate_sentence(counts, vocab))
    print("Perplexity:", perplexity(counts, vocab, corpus))

    print('Brown Corpus ====================')
    nltk.download('brown', quiet=True)
    sentences = brown.sents()
    corpus = [[s.lower() for s in sent] for sent in sentences]
    counts, vocab = train_bigram_model(corpus)
    print("Generated:", generate_sentence(counts, vocab))
    print("Perplexity:", perplexity(counts, vocab, corpus))


