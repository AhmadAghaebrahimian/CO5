

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Character size: ', len(text))
print('Vocabulary size: ', vocab_size)
