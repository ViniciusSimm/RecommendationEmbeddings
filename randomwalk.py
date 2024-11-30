import pandas as pd
import optuna
import numpy as np
import faiss
import ast
import networkx as nx
from torch_geometric.utils import from_networkx
import torch

dataframe = pd.read_csv('data/xiami/dataframe.csv')
dataframe['song_ids_order'] = dataframe['song_ids_order'].apply(ast.literal_eval)

dataframe = dataframe[dataframe['length'] > 3]

dataframe['last_item'] = dataframe['song_ids_order'].apply(lambda x: x.pop() if len(x) > 0 else None)

all_songs = set([item for sublist in dataframe['song_ids_order'] for item in sublist])

dataframe = dataframe[dataframe['last_item'].isin(all_songs)]

sequences = dataframe['song_ids_order']



G = nx.Graph()

for ids in sequences:
    # Garantir que 'ids' seja uma lista de strings
    ids = [str(id) for id in ids]
    # Conectar apenas IDs vizinhos na lista
    for i in range(len(ids) - 1):
        G.add_edge(ids[i], ids[i + 1])

print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# Exemplo de algumas arestas
edges_sample = list(G.edges())[:10]
print("Exemplos de arestas:", edges_sample)


# Criar mapeamentos entre IDs e índices
id_to_index = {id: idx for idx, id in enumerate(G.nodes())}
index_to_id = {idx: id for id, idx in id_to_index.items()}

# Relabelar os nós do grafo com índices inteiros
G_indexed = nx.relabel_nodes(G, id_to_index)

# Converter o grafo para um objeto Data do PyG
data = from_networkx(G_indexed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
print(device)

# ----------------------------------------------------------------------------- MODEL

def objective(trial):
    from torch_geometric.nn import Node2Vec

    vector_size = trial.suggest_categorical('vector_size', [64, 128, 256, 512, 1024])
    walk_length = trial.suggest_int('walk_length', 20, 50, step=5)  # Tamanho do passeio
    context_size = trial.suggest_int('context_size', 5, 19)  # Contexto para o modelo Skip-Gram
    walks_per_node = trial.suggest_int('walks_per_node', 1, 3)  # Número de passeios por nó
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 5)  # Número de amostras negativas

    print(f'This version: vector size: {vector_size}, walk_length: {walk_length}, context_size: {context_size}, walks_per_node: {walks_per_node}, num_negative_samples: {num_negative_samples}')

    # Modelo Node2Vec com os parâmetros sugeridos
    model = Node2Vec(
        data.edge_index,
        embedding_dim=vector_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=1,
        q=1,
        sparse=True
    ).to(device)

    # Otimizador
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    loss_values = []

    print('Start Training')

    # Simulating the training loop
    for epoch in range(1, 15):
        loss = train()
        loss_values.append(loss)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # Obter os embeddings finais
    embeddings = model.embedding.weight.data.cpu().numpy()

    # Mapear os embeddings aos IDs originais 
    embedding_dict = {index_to_id[idx]: embeddings[idx] for idx in range(len(embeddings))}

    def get_embedding(song_id):
        return embedding_dict.get(song_id)

    print('Calculating avg embeddings')

    # Adicione os embeddings às sequências
    dataframe['song_embeddings'] = sequences.apply(lambda seq: [get_embedding(song_id) for song_id in seq])

    dataframe['average_embedding'] = dataframe['song_embeddings'].apply(lambda embeddings: np.mean(embeddings, axis=0))

    def check_last_item_in_similar(row):
        last_item = str(row['last_item'])
        similar_ids = row['similar_song_ids']
        return last_item in similar_ids
    
    song_ids = [index_to_id[idx] for idx in range(len(index_to_id))]

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
    
    print('Calculating Score')

    score = calcular_score_final(embeddings, song_ids, dataframe)

    with open('randomwalk.txt', 'a') as f:
        row = f'vector_size={vector_size}, walk_length={walk_length}, context_size={context_size}, walks_per_node={walks_per_node}, num_negative_samples={num_negative_samples}, p=1, q=1, score={score}'
        f.write(row + '\n')

    return score


storage_name = "sqlite:///randomwalk.db"

study = optuna.create_study(direction='maximize', storage=storage_name, study_name="randomwalk", load_if_exists=True)
study.optimize(objective, n_trials=10)

print("Melhores hiperparâmetros: ", study.best_params)