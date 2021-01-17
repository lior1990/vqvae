import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from datasets.dataset import ImageDataset
from models.e2e_encoder import E2EEncoder
from models.vqvae import VQVAE
from summaries import TensorboardSummary

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--dataset",  type=str, required=True)
parser.add_argument("--model_path",  type=str, required=True)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)

dataset = ImageDataset(args.dataset, transform)
training_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
checkpoint = torch.load(os.path.join(utils.SAVE_MODEL_PATH, args.model_path), map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

encoder = E2EEncoder(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, args.embedding_dim)
encoder.to(device)
encoder.train()
"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, amsgrad=True)


results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():
    summary = TensorboardSummary("./results")

    for i in range(args.n_updates):
        x = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        z_ae = encoder(x)
        _, z_q, _, _, _ = model.vector_quantization(z_ae)
        x_hat = model.decoder(z_q)
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["n_updates"] = i

        summary.add_scalar("recon loss", recon_loss.item(), i)

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(encoder, results, hyperparameters, args.filename, prefix="encoder")
                if x.shape[0] > 3:
                    # plot different images every time
                    rand_indices_to_visualize = torch.randperm(x.shape[0])
                    x = x[rand_indices_to_visualize]
                    x_hat = x[rand_indices_to_visualize]
                summary.visualize_image(i, x, "real")
                summary.visualize_image(i, x_hat, "generated")

            print('Update #', i, 'Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]))


if __name__ == "__main__":
    train()
