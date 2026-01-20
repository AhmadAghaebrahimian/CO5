from whoosh import index
from II_index import index_dir
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin

ix = index.open_dir(index_dir)

with ix.searcher() as searcher:
    qp = MultifieldParser(["title", "body"], ix.schema, group=OrGroup) # group=AndGroup automatically make "fox quick" mean "fox AND quick" instead of the default OR behavior
    qp.add_plugin(FuzzyTermPlugin())# Typo-tolerant, uses Levenshtein distance.
    qp.add_plugin(PhrasePlugin()) # Enclose in quotes for exact match: "quick brown fox"

    while True:
        query_str = input("Enter search query (or 'exit' to quit): ")
        if query_str.lower() == "exit":
            break
        query = qp.parse(query_str)
        results = searcher.search(query, limit=100)
        print(f"Found {len(results)} result(s):")
        for hit in results:
            print(f'Confidence {hit.score} for Hit {hit}')



