"""
Main VLM vlm function.
"""

import torch
import torchvision
import os

from modeling.vlms.prompts import *
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VLMModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', vision_pretrained=True, proj_dim=512, proj_bias=False,
                 logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None, out_path=None, image_size=224,
                 projection=True, norm_features=True, modality="cxr", vlm_id=""):

        super().__init__()

        # Set attributes
        self.vision_type = vision_type
        self.modality = modality
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.out_path = out_path
        if self.out_path is not None:
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
        self.image_size = image_size
        self.projection = projection
        self.norm_features = norm_features
        self.vlm_id = vlm_id

        # Set base architecture: specialized models vs. clip
        if from_checkpoint:  # specialized architectures, pre-trained on target domain
            if self.modality == "fundus" or self.modality == "cxr":
                # FLAIR and CONVIRT ese Bio_ClinicalBERT as text encoder
                self.text_encoder_type = 'emilyalsentzer/Bio_ClinicalBERT'
                self.weights_path = weights_path
            elif self.modality == "histology":
                if self.vlm_id == "plip": # PLIP use CLIP as text encoder
                    self.text_encoder_type = 'openai/clip-vit-base-patch32'
                    self.weights_path = "vinid/plip"
                elif self.vlm_id == "conch":
                    self.text_encoder_type = "conch_ViT-B-16"
                    self.vision_type = "conch"
                    self.weights_path = weights_path

        else:  # using CLIP pre-training on natural image data
            if self.vision_type == "resnet":
                self.text_encoder_type = 'openai/clip-vit-base-patch32'
                self.weights_path = "clip_RN50"
                self.vision_dim, self.proj_dim = 2048, 1024
            elif self.vision_type == "vitb32":
                self.text_encoder_type = 'openai/clip-vit-base-patch32'
                self.weights_path = "openai/clip-vit-base-patch32"

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features, weights_path=self.weights_path)
        self.text_model = TextModel(text_encoder_type=self.text_encoder_type, proj_dim=self.proj_dim,
                                    proj_bias=self.proj_bias, projection=self.projection, norm=self.norm_features,
                                    weights_path=self.weights_path)

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # Load pretrained weights
        if self.vision_type not in ["conch"]:
            self.load_from_pretrained(self.weights_path)
        else:
            self.logit_scale = self.vision_model.logit_scale
            self.vision_model.logit_scale = None

        # Set model to device
        self.to(device).to(torch.float32)

    def load_from_pretrained(self, weights_path=None):

        if weights_path is None:
            print("Pre-trained weights path not specified")

        if self.modality == "histology":
            print('load model weight from:', self.weights_path)
            # Load state dict from open_clip
            state_dict = AutoModel.from_pretrained(self.weights_path, output_hidden_states=True).state_dict()
            # Correct key names
            for key in list(state_dict.keys()):
                if "text_model" in key:
                    state_dict[key.replace("text_model.", "text_model.model.")] = state_dict.pop(key)
                if "vision_model" in key:
                    state_dict[key.replace("vision_model.", "vision_model.model.")] = state_dict.pop(key)
            state_dict["vision_model.projection_head_vision.projection.weight"] =\
                state_dict.pop("visual_projection.weight")
            state_dict["text_model.projection_head_text.projection.weight"] =\
                state_dict.pop("text_projection.weight")
            strict = True
        elif self.weights_path == "clip_RN50":
            print('load model weight from clip library: ', "RN50")
            import clip
            model, preprocess = clip.load("RN50")
            self.vision_model.model = model.visual
            self.vision_model.projection_head_vision.projection = torch.nn.Identity()
            self.text_model.model = model.transformer
            self.text_model.projection_head_text.projection.weight = torch.nn.Parameter(model.text_projection.t())
            self.text_model.tokenizer = clip.tokenize
            self.text_model.positional_embedding = model.positional_embedding
            self.text_model.ln_final = model.ln_final
            self.text_model.token_embedding = model.token_embedding
            self.logit_scale = model.logit_scale
            self.to(device)
            return
        else:
            print('load model weight from:', weights_path)
            state_dict = torch.load(weights_path)
            strict = False

        self.load_state_dict(state_dict, strict=strict)

    def compute_text_embeddings(self, categories, disp=True):
        # Obtain text embeddings per class
        text_embeds_dict = {}
        text_prototypes = []
        text_labels = []

        # Determine number of prompts for ensemble or not
        if self.modality == "cxr":
            prompts = generate_prompt_cxr(100)
        if self.modality == "fundus":
            prompts = generate_prompt_fundus(categories)
        if self.modality == "histology":
            prompts = generate_prompt_histology(categories, model_id=self.text_encoder_type)

        for iKey in range(len(categories)):
            # Forwards prompts trough text encoder
            with torch.no_grad():
                descriptions = prompts[categories[iKey]]
                if disp:
                    print(descriptions)
                if self.weights_path == "clip_RN50":
                    text_token = self.text_model.tokenizer(descriptions).to(device)
                    text_embeds = self.text_model(text_token)
                elif "conch" in self.weights_path:
                    text_token = self.text_model.tokenizer.batch_encode_plus(
                        descriptions, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                        truncation=True, padding='max_length', return_tensors='pt').to(device)
                    text_token = torch.nn.functional.pad(text_token['input_ids'], (0, 1),
                                                         value=self.text_model.tokenizer.pad_token_id)
                    text_embeds = self.text_model(text_token)
                else:
                    text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True,
                                                           return_tensors='pt')
                    input_ids = text_token["input_ids"].to(device).to(torch.long)
                    attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                    text_embeds = self.text_model(input_ids, attention_mask)

            if len(text_embeds.shape) == 1:
                text_embeds = text_embeds.unsqueeze(0)

            # Add text prototypes
            text_prototypes.append(text_embeds.clone())
            text_labels.extend([iKey for i in range(text_embeds.shape[0])])

            # Get class prototypes
            text_embeds = text_embeds.mean(0).unsqueeze(0)
            text_embeds_dict[categories[iKey]] = text_embeds

        # Prepare output: classwise prototypes
        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        # Prepare output: text embedding points
        text_samples = torch.concat(text_prototypes, dim=0).cpu().numpy()
        dataset_text = {"embeddings": text_samples, "labels": text_labels}

        return text_embeds_dict, text_embeds, self.logit_scale, dataset_text


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True, weights_path=None):
        super().__init__()
        self.proj_dim = proj_dim
        self.vision_type = vision_type

        # Assert vision encoders
        if vision_type not in ['resnet', 'vitb32', "convnext", "conch"]:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type == "resnet":
            # Set pretrained weights from Imagenet and get model
            weights = 'IMAGENET1K_V1' if pretrained else None

            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "vitb32":
            # Set pretrained weights from Imagenet and get model
            weights = "openai/clip-vit-base-patch32" if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = AutoModel.from_pretrained(weights, output_hidden_states=True).vision_model
            # Set number of extracted features
            self.vision_dim = 768
            # Replace classifier by Identity layer
            self.model.heads = torch.nn.Identity()
        elif vision_type == "convnext":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.convnext_tiny(weights=weights)
            # Set number of extracted features
            self.vision_dim = 768
            # Replace classifier by Identity layer
            self.model.classifier = torch.nn.Flatten()
        elif vision_type == "conch":
            from conch.open_clip_custom import create_model_from_pretrained
            model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
            self.model = model.visual
            # Set number of extracted features
            self.vision_dim = 512
            self.logit_scale = model.logit_scale
            # Replace classifier by Identity layer
            projection, norm = False, True

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        if self.vision_type == "vitb32":
            embed = embed["pooler_output"]

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, text_encoder_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False,
                 projection=True, norm=True, weights_path=None):
        super().__init__()

        # Attributes
        self.text_encoder_type = text_encoder_type

        # Load text encoder from pretrained
        if "openai/clip" in text_encoder_type:
            self.model = AutoModel.from_pretrained(text_encoder_type, output_hidden_states=True).text_model
            # Set tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
            self.tokenizer.model_max_length = 77
            out_dim = 512
        elif "conch" in text_encoder_type:
            from conch.open_clip_custom import create_model_from_pretrained
            model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
            self.model = model.text
            # Replace classifier by Identity layer
            projection, norm = False, True
            out_dim = 512
            # Set tokenizer
            from conch.open_clip_custom import tokenize, get_tokenizer
            self.tokenizer = get_tokenizer()
            self.tokenizer.model_max_length = model.text.context_length
        else:
            self.model = AutoModel.from_pretrained(text_encoder_type, output_hidden_states=True)
            # Set tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
            self.tokenizer.model_max_length = 77
            out_dim = 768

        # Set projection head
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(out_dim, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def forward(self, input_ids, attention_mask=None):

        if "conch" in self.text_encoder_type:
            text = input_ids[:, :-1]
            text_latent, token_emb = self.model(text)
            # Compute projection from text embedding to multi-modal projection
            embed = self.projection_head_text(text_latent)
            return embed

        # Forwards trough text encoder
        if attention_mask is None:
            x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model(x.type(self.model.resblocks[0].attn.in_proj_weight.dtype))
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            embed = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
            return self.projection_head_text(embed)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine last feature layers to compute text embedding
        if "openai/clip" in self.text_encoder_type:
            embed = output["pooler_output"]
        elif "Bio_ClinicalBERT" in self.text_encoder_type:
            last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                              output['hidden_states'][-1]])
            embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Compute projection from text embedding to multi-modal projection
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        if projection:
            self.norm_modality = False
        else:
            self.norm_modality = norm
        self.norm_projection = norm
        self.projection = layer
        self.last_features = None

    def forward(self, x):

        # Storing last features before projection
        self.last_features = x

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x