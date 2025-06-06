import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def adapt(features, labels, model):

    # Fit the fast linear probing SS-Text solver
    z, model = ss_text_solver(features, labels, model)

    return z, model


def ss_text_solver(features, labels, model):

    # Number of samples
    N = labels.shape[0]

    # Compute new class centers
    with torch.no_grad():

        # Labels to ohe
        affinity_labeled = torch.nn.functional.one_hot(labels).float()

        # Compute new class centers (visual)
        tau = (1 / model.adapter.logit_scale.exp().item()) # temperature scale
        vision_mu = torch.einsum('ij,ik->jk', affinity_labeled, features) / tau

        # Extract zero-shot prototypes
        text_mu = model.adapter.prototypes.data.clone().to("cpu")

        # Adjust lambda based on amount of support data available
        lambda_text = torch.tensor((1 / (N)))

        # Avoid NaN
        lambda_text = lambda_text.clamp(min=1e-3)

        # Compute optimum weights via ss-text solver
        new_mu = (1/N) * (1/lambda_text) * vision_mu + text_mu

    # Set Adapter
    model.adapter.prototypes.data = new_mu.to(device)

    # Compute predictions
    with torch.no_grad():
        z = torch.softmax(model(features.to(device)), -1)

    return z, model