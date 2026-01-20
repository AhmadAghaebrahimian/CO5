import os
import pickle
import shutil
from tqdm import tqdm
from whoosh import index
from I_fetch_pubmed import medline_folder
from whoosh.fields import Schema, TEXT, ID


# Schema
schema = Schema(
    id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    body=TEXT(stored=True)
)

# Index
index_dir = "pubmed_index"


def get_index():
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)
    writer = ix.writer()
    for pkl_file in os.listdir(medline_folder):
        pkl_obj = pickle.load(open(os.path.join(medline_folder, pkl_file), 'rb'))
        print('Indexing %s ...'%pkl_file)
        for idx in tqdm(pkl_obj):
            writer.add_document(id=str(idx), title=pkl_obj[idx][0], body=pkl_obj[idx][1])
    writer.commit()

if __name__ == "__main__":
    get_index()

