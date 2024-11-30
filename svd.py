import pandas as pd
import optuna
import numpy as np
import faiss
import ast

dataframe = pd.read_csv('data/xiami/dataframe.csv')

dataframe['song_ids_order'] = dataframe['song_ids_order'].apply(ast.literal_eval)

dataframe = dataframe[dataframe['length'] > 3]

dataframe['last_item'] = dataframe['song_ids_order'].apply(lambda x: x.pop() if len(x) > 0 else None)

all_songs = set([item for sublist in dataframe['song_ids_order'] for item in sublist])

dataframe = dataframe[dataframe['last_item'].isin(all_songs)]

sequences = dataframe['song_ids_order']

song_to_idx = {song_id: idx for idx, song_id in enumerate(all_songs)}

idx_to_song = {idx: song_id for song_id, idx in song_to_idx.items()}

num_songs = len(all_songs)

# ----------------------------------------------------------------------------- MODEL

def objective(trial):

    # Hiperparâmetros que serão ajustados
    vector_size = trial.suggest_categorical('vector_size', [128, 256, 512, 1024])
    window = trial.suggest_int('window', 3, 15)
    
    cooccurrence_matrix = create_cooccurrence_matrix(num_songs, song_to_idx, sequences, window)

    from sklearn.decomposition import TruncatedSVD

    # Defina o número de componentes

    # Aplique o SVD
    svd = TruncatedSVD(n_components=vector_size, random_state=42, n_iter=20)
    embeddings = svd.fit_transform(cooccurrence_matrix)

    embedding_df = pd.DataFrame(embeddings, index=[idx_to_song[idx] for idx in range(num_songs)])
    embedding_df.index.name = 'song_id'
    embedding_df.reset_index(inplace=True)

    song_embeddings = {row['song_id']: row[embedding_df.columns[1:]].values for _, row in embedding_df.iterrows()}

    def get_embedding(song_id):
        return song_embeddings.get(song_id, np.zeros(vector_size))

    # Adicione os embeddings às sequências
    dataframe['song_embeddings'] = sequences.apply(lambda seq: [get_embedding(song_id) for song_id in seq])

    # Opcional: Calcule um embedding médio para cada sequência
    dataframe['average_embedding'] = dataframe['song_embeddings'].apply(lambda embeddings: np.mean(embeddings, axis=0))

    song_ids = [idx_to_song[idx] for idx in range(len(idx_to_song))]

    # Processo para identificar o score final (substitua pelo seu método)
    score = calcular_score_final(embeddings, song_ids, dataframe)

    with open('svd.txt', 'a') as f:
        row = f'vector_size={vector_size}, window={window}, score={score}'
        f.write(row + '\n')
    
    return score

def get_average_embedding(song_ids, model):
    embeddings = []
    for song_id in song_ids:
        song_id_str = str(song_id)
        if song_id_str in model.wv:
            embeddings.append(model.wv[song_id_str])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)
    
def check_last_item_in_similar(row):
    last_item = str(row['last_item'])
    similar_ids = row['similar_song_ids']
    return last_item in similar_ids

def calcular_score_final(embeddings, song_ids, dataframe):
    
    # Preparar embeddings para comparação
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # mean_vector_normalized = mean_vector / np.linalg.norm(mean_vector)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Usando Inner Product para similaridade cosseno
    index.add(embeddings)
    
    average_embeddings = np.stack(dataframe['average_embedding'].values).astype('float32')
    average_embeddings = average_embeddings / np.linalg.norm(average_embeddings, axis=1, keepdims=True)
    
    k = 10
    distances, indices = index.search(average_embeddings, k)
    similar_song_ids = [[song_ids[idx] for idx in neighbors] for neighbors in indices]
    dataframe['similar_song_ids'] = similar_song_ids
    
    dataframe['is_last_item_in_similar'] = dataframe.apply(check_last_item_in_similar, axis=1)

    counts = dataframe.groupby('is_last_item_in_similar').size()

    if True in counts:
        # Calcular a porcentagem de True
        percentage_true = (counts[True] / counts.sum()) * 100
    else:
        # Se não houver nenhum True, a porcentagem é 0
        percentage_true = 0

    return percentage_true

def create_cooccurrence_matrix(num_songs, song_to_idx, sequences, window_size):
    from scipy.sparse import lil_matrix
    cooccurrence_matrix = lil_matrix((num_songs, num_songs), dtype=np.float64)

    # Preencha a matriz de coocorrência
    for seq in sequences:
        seq_indices = [song_to_idx[song_id] for song_id in seq]
        for i, idx_target in enumerate(seq_indices):
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(seq_indices))
            context_indices = seq_indices[start:i] + seq_indices[i+1:end]
            for idx_context in context_indices:
                cooccurrence_matrix[idx_target, idx_context] += 1

    # Converta a matriz para formato CSR para uso no SVD
    cooccurrence_matrix = cooccurrence_matrix.tocsr()

    return cooccurrence_matrix

storage_name = "sqlite:///svd.db"

study = optuna.create_study(direction='maximize', storage=storage_name, study_name="svd", load_if_exists=True)
study.optimize(objective, n_trials=25)

print("Melhores hiperparâmetros: ", study.best_params)