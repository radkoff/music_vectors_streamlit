from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

here = Path(__file__).parent

def embed(albums: pd.DataFrame, n_components=16):
    """
    Uses word descriptions from the top 5k albums on rateyourmusic to come up with an embedding model.
    Returns a 2d array of vectors for the given `albums` dataframe
    """

    # https://www.kaggle.com/datasets/tobennao/rym-top-5000/data
    rym_dump = pd.read_csv(here / '../data/rym_clean1.csv')
    rym_dump = rym_dump.reset_index(drop=True)

    # Verify all albums are found in the rym top 5k
    target_indices = []
    for _, album in albums.iterrows():
        matches = rym_dump[
            (rym_dump.artist_name.str.lower() == album.artist.lower()) &
            (rym_dump.release_name.str.lower() == album.album.lower())
        ]
        if len(matches) != 1:
            raise ValueError(f"No match found for {album}")
        target_indices.append(matches.iloc[0].name)

    unique_desc = set()

    album_descs = []
    for desc_str in rym_dump.descriptors:
        album_desc = []
        for d in desc_str.split(','):
            clean_d = d.strip().lower()
            unique_desc.add(clean_d)
            album_desc.append(clean_d)
        album_descs.append(album_desc)

    vocab = list(unique_desc)
    word_to_idx = dict(zip(vocab, range(len(unique_desc))))

    matrix = np.zeros((len(rym_dump), len(vocab)), dtype=float)
    for i, descs in enumerate(album_descs):
        for d in descs:
            matrix[i, word_to_idx[d]] += 1.0

    # Tried TruncatedSVD, PCA, and UMAP. UMAP does the best I think
    umap_model = UMAP(n_neighbors=15, n_components=n_components, min_dist=0.0, metric='cosine')
    all_emb = umap_model.fit_transform(matrix)
    # 0-1 normalization as post-processing
    all_emb = all_emb - all_emb.min(axis=0)
    all_emb = all_emb / all_emb.max(axis=0)

    return all_emb[target_indices, :]
