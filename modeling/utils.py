import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_vision_features(vision_model, loader, augmentations=False):

    # Set repetitions based on required augmentations
    reps = 10 if augmentations else 1

    # Loop to extract features
    refs_ds_rep, feats_ds_rep = [], []
    for irep in range(reps):
        refs_ds, feats_ds = [], []
        for step, batch in enumerate(loader):
            print("  Batch {ii}/{II}".format(
                ii=step + 1, II=len(loader)), end="\n")

            # Retrieve images and labels
            images = batch["image"].to(device).to(torch.float32)
            refs_batch = batch["label"].to(device).to(torch.float32)

            # Forward predictions
            with torch.no_grad():
                feats = vision_model(images.to(device).float())

            if type(feats) is tuple:
                feats = feats[0]

            # Store labels and predictions
            refs_ds.append(refs_batch.detach().cpu().numpy()), feats_ds.append(feats.detach().cpu().numpy())

        # Concatenate features and refs
        refs_ds = np.concatenate(refs_ds)
        feats_ds = np.concatenate(feats_ds, axis=0)

        refs_ds_rep.append(np.expand_dims(refs_ds, -1)), feats_ds_rep.append(np.expand_dims(feats_ds, -1))

    # Concatenate augmentations
    refs_ds_rep = np.squeeze(np.concatenate(refs_ds_rep, axis=-1))
    feats_ds_rep = np.squeeze(np.concatenate(feats_ds_rep, axis=-1))

    return feats_ds_rep, refs_ds_rep


def predict_from_features(adapter, feats_ds, bs=512, act=True, epsilon=1.0):
    preds, idx = [], 0
    while idx <= feats_ds.shape[0]:

        # Retrieve features
        x = feats_ds[idx:idx+bs, :].to(device).to(torch.float32)

        # Forward predictions
        with torch.no_grad():
            pred = adapter(x)
            if act:
                pred = torch.softmax(pred / epsilon, dim=-1)

        # Store labels and predictions
        preds.append(pred.detach().cpu())

        # Update iterator
        idx += bs

    # Concatenate predictions
    preds = torch.cat(preds, axis=0)

    return preds
