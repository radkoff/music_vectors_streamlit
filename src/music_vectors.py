from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial import distance
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import streamlit as st
import umap

project_root = Path(__file__).parent.parent
if project_root.as_posix() not in sys.path:
    sys.path.append(project_root.as_posix())

from src import rym_descriptors, utils

EMB_PATH = project_root / 'data/blog_music_vectors.pkl'
RANDOM_STATE = 0

# Query params have been added to support all the necessary views for the blog post --
# entity_choice, skip_data_sources, nn, skip_dim_reduction, cluster, mobile

essentia_col_names = [
    'danceability', 'gender', 'mood_acoustic', 'mood_aggressive', 'mood_electronic', 'mood_happy', 'mood_party',
    'mood_relaxed', 'mood_sad', 'mirex_mood1', 'mirex_mood2', 'mirex_mood3', 'mirex_mood4', 'mirex_mood5', 'timbre',
    'tonal', 'instrumental'
]

@st.cache_data
def get_data(entity_choice: str, data_sources: list[str], distance_matrix: bool):
    """
    Returns a tuple of (DataFrame of track or album entities, vectors)
    If `distance_matrix` is true, the returned vectors are pairwise distances instead of raw feature vectors.
    The @st.cache_data decorator means streamlit with only have to run this once per argument combination.
    """
    track_entities = pd.read_pickle(EMB_PATH)
    track_entities = track_entities[~track_entities.genre_embedding.isna()]
    if entity_choice == 'Tracks':
        entities = track_entities.copy()
    elif entity_choice == 'Albums':
        entities = utils.tracks_to_albums(track_entities, embedding_cols={'genre_embedding', 'essentia_embedding'})

    embeddings = np.zeros((len(entities), 0))
    embedding_names = []  # For debugging
    embedding_sizes = []  # For equally scaling each data source

    distance_matrices = []

    def _distance(features):
        return distance.squareform(distance.pdist(features, 'euclidean'))

    if 'Genre Classification Embeddings' in data_sources or 'Aggregated Genre Embeddings' in data_sources:
        genre_emb = np.array(entities.genre_embedding.values.tolist(), dtype=float)
        embeddings = np.concatenate([embeddings, genre_emb], axis=1)
        embedding_names += [f'genre{i}' for i in range(genre_emb.shape[1])]
        embedding_sizes.append(genre_emb.shape[1])
        distance_matrices.append(_distance(genre_emb))
    if 'Essentia Mood Features' in data_sources or 'Aggregated Essentia Features' in data_sources:
        essentia_emb = np.array(entities.essentia_embedding.values.tolist(), dtype=float)
        # A few features seem to be messed up, with values of 0/0/1 (it's almost always all three like that, for about
        # ~40% of tracks that I tested, with no identifiable pattern) and it throws off clustering
        cleanup_cols = [0, 3, 16]
        essentia_emb = np.delete(essentia_emb, cleanup_cols, axis=1)

        embeddings = np.concatenate([embeddings, essentia_emb], axis=1)
        col_names = [val for idx, val in enumerate(essentia_col_names) if idx not in cleanup_cols]
        embedding_names += col_names
        embedding_sizes.append(len(col_names))
        distance_matrices.append(_distance(essentia_emb))
    if 'RateYourMusic Descriptors' in data_sources:
        rym_emb = rym_descriptors.embed(entities, n_components=16)
        embeddings = np.concatenate([embeddings, rym_emb], axis=1)
        embedding_names += [f'rym{i}' for i in range(rym_emb.shape[1])]
        embedding_sizes.append(rym_emb.shape[1])
        distance_matrices.append(_distance(rym_emb))

    name_attr = 'name' if entity_choice == 'Tracks' else 'album'
    entities['label'] = entities.apply(
        lambda ent: f'{ent.artist} - {ent[name_attr]}', axis=1
    )

    # Ensure each data source contributes equally to the output vectors
    if distance_matrix:
        print(f"Returning distance matrix of shape {distance_matrices[0].shape}")
        return entities, np.mean(distance_matrices, axis=0)
    else:
        # If we're not precomputing a distance matrix, another way to (approximately) normalize each source is
        # dividing by inverse source dimensionality. This will only be used for PCA
        total = sum(embedding_sizes)
        idx = 0
        for emb_size in embedding_sizes:
            embeddings[:, idx:idx + emb_size] *= total / (len(embedding_sizes) * emb_size)
            idx += emb_size
        print(f"Embeddings of shape {embeddings.shape}")

        # Optional debug chart

        # index = entities.tag_music_brainz_recording_id if entity_choice == 'Tracks' else entities.album
        # def color_classification(col):
        #     colors = []
        #     for v in col.values:
        #         if np.isclose(v, 0):
        #             colors.append('background-color: red')
        #         elif np.isclose(v, 1):
        #             colors.append('background-color: green')
        #         else:
        #             colors.append('')
        #     return colors
        # alpha_sort = np.argsort(entities.label)
        # st.table(
        #     pd.DataFrame(pre_scaled_emb[alpha_sort.values], columns=embedding_names,
        #                  index=entities.iloc[alpha_sort.values].label)
        #     .style.background_gradient(axis=0).apply(color_classification).format('{:.2f}')
        # )

        return entities, embeddings

@st.cache_data
def get_data_reduced(
        entity_choice: str, data_sources: list[str], dim_reduction: str, include_clusters: bool, perplexity: int,
        n_neighbors: int
):
    """
    Prepares a DataFrame of track or album entities, apt for scatterplots with X and Y coordinate projections and
    optionally cluster IDs.

    If we're using UMAP or TSNE, we'll compute distance matrices for each data source and average them
    If we're using PCA first, we'll scale each vector by its data source dimensionality, thus the return val here
      is still the original feature matrix.
    The @st.cache_data decorator means streamlit with only have to run this once per argument combination.
    """
    using_distance_matrices = 'PCA' not in dim_reduction
    dist_metric = 'precomputed' if using_distance_matrices else 'euclidean'
    entities, embeddings = get_data(entity_choice, data_sources, distance_matrix=using_distance_matrices)

    umap_min_dist = 0.05
    if dim_reduction == 't-SNE':
        projection = TSNE(
            random_state=RANDOM_STATE, perplexity=perplexity, n_jobs=1, metric=dist_metric, init='random',
        ).fit_transform(embeddings)
    elif dim_reduction == 'PCA':
        projection = PCA(random_state=RANDOM_STATE, n_components=2).fit_transform(embeddings)
    elif dim_reduction == 'PCA → t-SNE':
        projection = PCA(random_state=RANDOM_STATE, n_components=8).fit_transform(embeddings)
        projection = TSNE(random_state=RANDOM_STATE, perplexity=perplexity, n_jobs=1).fit_transform(projection)
    elif dim_reduction == 'UMAP':
        projection = umap.UMAP(
            random_state=RANDOM_STATE, min_dist=umap_min_dist, n_neighbors=n_neighbors, n_jobs=1, metric=dist_metric
        ).fit_transform(embeddings)
    elif dim_reduction == 'PCA → UMAP':
        projection = PCA(random_state=RANDOM_STATE, n_components=8).fit_transform(embeddings)
        projection = umap.UMAP(
            random_state=RANDOM_STATE, min_dist=umap_min_dist, n_neighbors=n_neighbors, n_jobs=1
        ).fit_transform(projection)

    projection = sklearn.preprocessing.minmax_scale(projection)
    entities['x'] = projection[:, 0]
    entities['y'] = projection[:, 1]
    # For debugging
    entities['rounded_emb'] = embeddings.round(2).tolist()

    if include_clusters:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=5, store_centers='centroid')
        clusters = clusterer.fit_predict(projection)

        # Instead of allowing outliers, assign them to the closest group centroid
        n_valid_assignments = (clusters != -1).sum()
        if n_valid_assignments > 0 and n_valid_assignments < len(clusters):
            for i in range(len(clusters)):
                if clusters[i] == -1:
                    distances = np.apply_along_axis(
                        lambda centroid: distance.euclidean(centroid, projection[i]), 1, clusterer.centroids_
                    )
                    clusters[i] = np.argmin(distances)
        entities['cluster'] = list(map(str, clusters))

    return entities


possible_entities = ['Tracks', 'Albums']
entity_choice = st.query_params.get('entity_choice') or st.radio('Entities', options=possible_entities, horizontal=True)
if entity_choice == 'Tracks':
    data_source_options = ['Genre Classification Embeddings', 'Essentia Mood Features']
elif entity_choice == 'Albums':
    data_source_options = ['Aggregated Genre Embeddings', 'Aggregated Essentia Features', 'RateYourMusic Descriptors']
else:
    raise ValueError(f"entity_choice not in {possible_entities}")

if st.query_params.get('skip_data_sources', False):
    data_sources = ['Genre Classification Embeddings']
else:
    data_sources = set(st.multiselect('Data sources', data_source_options, data_source_options))

if len(data_sources) == 0:
    st.text('Select at least one data source')
elif st.query_params.get('nn', False):
    # Generally this is script is for scatterplots, but for one section we want a table with nearest neighbors
    entities, embeddings = get_data(entity_choice, data_sources, False)
    # Sort alphabetically by artist for a better UX
    order = np.argsort(entities.artist).values
    entities = entities.iloc[order]
    embeddings = embeddings[order]

    st.header(f'Nearest neighbors to...')
    initial_idx = 12
    query_idx = st.selectbox(
        label=entities.iloc[initial_idx].label,
        label_visibility='collapsed',
        options=list(range(len(entities))),
        format_func=lambda idx: entities.iloc[idx].label,
        index=initial_idx
    )

    tree = KDTree(embeddings)
    dist, neighbor_ind = tree.query([embeddings[query_idx]], k=9)
    results = pd.DataFrame([
        dict(Track=entities.iloc[idx].label, Distance=d)
        for d, idx in zip(dist[0][1:], neighbor_ind[0][1:])
    ])
    st.table(results)
else:
    dim_reduction = st.radio(
        'Dimensionality reduction', options=['t-SNE', 'PCA', 'UMAP', 'PCA → t-SNE', 'PCA → UMAP'], index=0,
        horizontal=True
    )
    if 't-SNE' in dim_reduction:
        opts = [15, 30, 45, 59] if entity_choice == 'Tracks' else [8, 12, 15, 20]
        if st.query_params.get('skip_dim_reduction', False):
            perplexity = opts[1]
        else:
            perplexity = st.select_slider('Perplexity', options=opts, value=opts[1])
    else:
        perplexity = None
    if 'UMAP' in dim_reduction:
        opts = [10, 15, 20, 25] if entity_choice == 'Tracks' else [5, 8, 12, 15]
        if st.query_params.get('skip_dim_reduction', False):
            n_neighbors = opts[2]
        else:
            n_neighbors = st.select_slider('Num neighbors', options=opts, value=opts[1])
    else:
        n_neighbors = None

    include_clusters = st.query_params.get('cluster', False)

    entities = get_data_reduced(
        entity_choice, data_sources, dim_reduction, include_clusters, perplexity, n_neighbors
    )

    name_tooltip = alt.Tooltip('name', title='track') if entity_choice == 'Tracks' else 'album'
    tooltip = ['artist', name_tooltip, 'genre', alt.Tooltip('year:T', format='%Y')]
    if include_clusters:
        tooltip.append('cluster')

    mobile = st.query_params.get('mobile', False)
    color_encoding = alt.Color('artist:N').legend(orient='bottom', columns=2) if mobile else 'artist:N'
    encode_kwargs = dict(color=color_encoding, tooltip=tooltip)
    if include_clusters:
        encode_kwargs['shape'] = 'cluster:N'

    alt.themes.enable('googlecharts')

    selection = alt.selection_point(
        fields=['artist'], bind='legend', empty=False, on=('click' if mobile else 'mouseover')
    )
    scatter = alt.Chart(entities, height=(500 if mobile else 400)).mark_point(filled=True).encode(
        alt.X('x').axis(None).scale(domain=(-.1, 1.1)),
        alt.Y('y').axis(None).scale(domain=(-.1, 1.1)),
        size=alt.condition(selection, alt.value(350), alt.value(150)),
        **encode_kwargs
    ).interactive().add_params(
        selection
    )

    st.altair_chart(scatter, use_container_width=True)
