import torch

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adapter(torch.nn.Module):
    def __init__(self, initial_prototypes, logit_scale, adapter="ZS"):
        super().__init__()

        # Init
        self.adapt_strategy = adapter
        self.logit_scale = torch.tensor(logit_scale)
        self.logit_scale.requires_grad = False
        self.text_embeddings_avg = initial_prototypes

        # Set strategy for classifier head initialization
        self.init = "random" if ("RI" in adapter) else "zero_shot"

        # Set classifier
        if "LP++" in adapter:
            self.adapter = LPpp(self.text_embeddings_avg)
        elif "TIPAd" in adapter:
            self.adapter = TIPAd(self.text_embeddings_avg)
        else:
            self.adapter = LinearProbeHead(self.text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()

    def forward(self, x):

        # Forward classifier
        out = self.adapter(x)

        return out

    def reset(self):

        # Set classifier
        self.adapter = LinearProbeHead(self.text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()


class LinearProbeHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale, init="zero_shot"):
        super().__init__()
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False
        self.init = init
        self.zero_shot_prot = zero_shot_prot.clone()

        if init == "zero_shot":
            self.prototypes = zero_shot_prot.clone()
        else:
            self.prototypes = torch.nn.init.kaiming_normal_(torch.empty(zero_shot_prot.shape))

        # Trainable parameters
        self.prototypes = torch.nn.Parameter(self.prototypes)

        # Keep temperature scaling as in pre-training
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False

    def forward(self, features):

        # Get trained prototype
        prototypes = self.prototypes.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits