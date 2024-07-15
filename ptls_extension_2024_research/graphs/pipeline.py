from ptls_extension_2024_research.graphs.static_models.gnn import GraphSAGE
from ptls_extension_2024_research.graphs.static_models.model import MainGNNModel

import torch

embedding_dim = 64
n_layers = 3


gnn_model = MainGNNModel(n_users, n_items, embedding_dim, train_user_embeddings=True, train_item_embeddings=True,
                 gnn_name='graphsage')


def init_gnn_model(n_clients, n_, ):



def train():

    model = GraphSAGE(embedding_dim, num_items, embedding_dim, n_layers)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Train LightGCN
    for epoch in range(num_epochs):
        model.train()
        user_ids = torch.randint(0, num_users, (1024,))
        pos_item_ids = torch.randint(0, num_items, (1024,))
        neg_item_ids = torch.randint(0, num_items, (1024,))

        loss = model.bpr_loss(user_ids, pos_item_ids, neg_item_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
