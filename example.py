import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from models import GNN


def train(loader, optimizer, criterion, model):
    overall_loss = []
    for batch in loader:
        model.train()
        optimizer.zero_grad()
        out, mc, o, s = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss += mc + o
        loss.backward()
        optimizer.step()
        overall_loss.append(loss)
    return torch.mean(torch.stack(overall_loss))

def evaluate(model, loader):
    model.eval()
    correct = 0
    for batch in loader:
        out, mc, o, s = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
    return correct / len(loader.dataset)

def get_data(name, train_ratio=0.6, test_ratio=0.2):
    dataset = TUDataset("TUDataset", name=name)
    dataset = dataset.shuffle()
    train_idx = int(len(dataset) * train_ratio)
    test_idx = int(len(dataset) * (1.0 - test_ratio))
    train_dataset = dataset[:train_idx]
    val_dataset = dataset[train_idx: test_idx]
    test_dataset = dataset[test_idx:]
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":

    # Load dataset
    train_dataset, val_dataset, test_dataset = get_data(name="MUTAG")

    train_loader = DataLoader(train_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Find max
    max_num_nodes = []
    for batch in train_dataset:
        max_num_nodes.append(batch.x.size(0))
    max_num_nodes = max(max_num_nodes)

    # Load model
    model = GNN(
        num_nodes=max_num_nodes,
        num_node_features=train_dataset.num_node_features,
        num_classes=train_dataset.num_classes,
        hidden_dim=[64, 64],
        mlp_hidden_dim=32,
        mu=0.1,
        cluster_ratio=0.25
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    for epoch in range(1, 200):
        loss = train(train_loader, optimizer, criterion, model)

    # Evaluate model
    test_acc = evaluate(model, test_loader)
    print("Performance accuracy is {:.4f}".format(test_acc))
