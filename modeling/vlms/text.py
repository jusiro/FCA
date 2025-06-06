import torch

from .prompts import generate_prompt_histology, generate_prompt_fundus, generate_prompt_cxr

from conch.open_clip_custom import tokenize, get_tokenizer

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_text_prototypes(model, categories, vlm_id=None, disp=False):
    # Obtain text embeddings per class
    text_embeds_dict = {}
    text_prototypes, text_labels = [], []

    # Determine number of prompts for ensemble or not
    if vlm_id == "conch":
        prompts = generate_prompt_histology(categories)
    elif vlm_id == "flair":
        prompts = generate_prompt_fundus(categories)
    elif vlm_id == "convirt":
        prompts = generate_prompt_cxr(100)

    for iKey in range(len(categories)):
        # Forwards prompts trough text encoder
        with torch.no_grad():
            descriptions = prompts[categories[iKey]]

            # Display prompts
            if disp:
                print(descriptions)

            if vlm_id == "conch":
                # Set tokenizer
                tokenizer = get_tokenizer()
                tokenizer.model_max_length = model.text.context_length

                # Produce text embeddings
                text_token = tokenizer.batch_encode_plus(
                    descriptions, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                    truncation=True, padding='max_length', return_tensors='pt').to(device)
                text_token = torch.nn.functional.pad(text_token['input_ids'], (0, 1),
                                                     value=tokenizer.pad_token_id)
                text = text_token[:, :-1]
                text_embeds, _ = model.text(text)

            elif vlm_id == "flair":
                text_token = model.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = model.text_model(input_ids, attention_mask)

            elif vlm_id == "convirt":
                text_token = model.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = model.text_model(input_ids, attention_mask)

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

    return text_embeds