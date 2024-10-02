import numpy as np

def tracks_to_albums(track_df, embedding_cols: set[str]=None):
    """
    Merges track entities into album entities

    :param track_df: DataFrame of track metadata and vectors
    :param embedding_cols: set of string column names to aggregate via averaging
    :return: DataFrame of album metadata and vectors
    """
    track_df['merge_artist'] = track_df.artist.str.lower().str.strip()
    track_df['merge_album'] = track_df.album.str.lower().str.strip()
    merged = track_df.groupby(['merge_artist', 'merge_album']).head(1)

    if len(embedding_cols) > 0:
        # Maps from the embedding col name to a dict from (merge_artist, merge_album) tuple to the new vectors
        new_embeddings = dict([(col, dict()) for col in embedding_cols])

        for col in embedding_cols:
            for idx, track in track_df.iterrows():
                key = (track.merge_artist, track.merge_album)
                vector = np.array(track[col])
                if key in new_embeddings[col]:
                    new_embeddings[col][key] = np.mean([new_embeddings[col][key], vector], axis=0)
                else:
                    new_embeddings[col][key] = vector
        for idx, album in merged.copy().iterrows():
            key = (album.merge_artist, album.merge_album)
            for col in embedding_cols:
                merged.at[idx, col] = new_embeddings[col][key]

    merged.loc[:, 'name'] = merged['album'].copy()
    return merged.drop(['merge_artist', 'merge_album'], axis=1)
