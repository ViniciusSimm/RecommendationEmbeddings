import pandas as pd
import optuna
import numpy as np
import faiss
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import ast

dataframe = pd.read_csv('data/xiami/dataframe.csv')

dataframe['song_ids_order'] = dataframe['song_ids_order'].apply(ast.literal_eval)

dataframe = dataframe[dataframe['length'] > 3]

dataframe['last_item'] = dataframe['song_ids_order'].apply(lambda x: x.pop() if len(x) > 0 else None)

all_songs = set([item for sublist in dataframe['song_ids_order'] for item in sublist])

dataframe = dataframe[dataframe['last_item'].isin(all_songs)]

sequences = dataframe['song_ids_order']

# ----------------------------------------------------------------------------- MODEL

from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    '''Callback para registrar a perda após cada época.'''
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_previous_step
        self.loss_previous_step = loss
        self.losses.append(loss_now)
        print(f'Perda após época {self.epoch}: {loss_now}')
        self.epoch += 1


class EarlyStoppingCallback(CallbackAny2Vec):
    '''Callback para early stopping baseado em uma melhoria percentual na perda.'''
    def __init__(self, epoch_logger, patience=3, min_percent_improvement=0.01):
        self.epoch_logger = epoch_logger
        self.patience = patience  # número de épocas sem melhoria
        self.min_percent_improvement = min_percent_improvement  # percentual mínimo de melhoria
        self.best_loss = float('inf')  # inicializar com um valor alto
        self.counter = 0  # contar épocas sem melhoria

    def on_epoch_end(self, model):
        loss_now = self.epoch_logger.losses[-1]  # Obter a última perda registrada pelo EpochLogger
        
        # Checar se houve uma melhoria percentual significativa
        if self.best_loss == float('inf'):
            improvement = float('inf')
        else:
            improvement = (self.best_loss - loss_now) / self.best_loss

        print(f'Melhoria percentual: {improvement * 100:.2f}%')

        if improvement > self.min_percent_improvement:
            self.best_loss = loss_now
            self.counter = 0  # Resetar o contador de épocas sem melhoria
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            
            # Parar se o número de épocas sem melhoria exceder a paciência
            if self.counter >= self.patience:
                print(f'Early stopping ativado na época {self.epoch_logger.epoch}')
                model.running_training = False  # Isso interrompe o treinamento

def objective(trial):

    epoch_logger = EpochLogger()
    early_stopping = EarlyStoppingCallback(epoch_logger=epoch_logger, patience=3, min_percent_improvement=0.01)

    # Hiperparâmetros que serão ajustados
    vector_size = trial.suggest_categorical('vector_size', [128, 256, 512, 1024])
    window = trial.suggest_int('window', 3, 15)
    alpha = trial.suggest_float('alpha', 1e-4, 1e-1, log=True)
    min_alpha = trial.suggest_float('min_alpha', 1e-4, 1e-2, log=True)
    negative = trial.suggest_int('negative', 5, 20)
    sample = trial.suggest_float('sample', 0.0001, 0.01, log=True)
    hs = trial.suggest_categorical('hs', [0, 1])
    sg = 0
    
    # Treinamento do modelo Word2Vec
    model = Word2Vec(
        sequences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        sg=sg,
        hs=hs,
        negative=negative,
        epochs=25,
        sample=sample,
        alpha=alpha,
        min_alpha=min_alpha,
        compute_loss=True,
        callbacks=[epoch_logger, early_stopping],  # Callback para log
        seed=42
    )

    # Processo para identificar o score final (substitua pelo seu método)
    score = calcular_score_final(model, dataframe)

    with open('word2vec.txt', 'a') as f:
        row = f'vector_size={vector_size}, window={window}, alpha={alpha}, min_alpha={min_alpha}, negative={negative}, sg={sg}, score={score}'
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

def calcular_score_final(model, dataframe):
    # Gerar embeddings médios
    dataframe['average_embedding'] = dataframe['song_ids_order'].apply(lambda x: get_average_embedding(x, model))
    
    # Preparar embeddings para comparação
    song_ids = list(model.wv.index_to_key)
    embeddings = np.array([model.wv[song_id] for song_id in song_ids]).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
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

storage_name = "sqlite:///word2vec_sg0.db"

study = optuna.create_study(direction='maximize', storage=storage_name, study_name="word2vec_sg0", load_if_exists=True)
study.optimize(objective, n_trials=50)

print("Melhores hiperparâmetros: ", study.best_params)