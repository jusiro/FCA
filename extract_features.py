"""
Main function for logit extraction using CLIP models
"""

import argparse
import torch
import os
import time
import datetime
import numpy as np

from torch.multiprocessing import set_start_method

from conch.open_clip_custom import create_model_from_pretrained

from modeling.utils import extract_vision_features, predict_from_features
from modeling.adapters.models import Adapter
from modeling.vlms.configs import get_model_config
from modeling.vlms.text import get_text_prototypes

from modeling.vlms.model import VLMModel

from data.configs import get_task_setting, get_experiment_setting
from data.dataloader import set_loader
from local_data.constants import *
from modeling.vlms.constants import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # %---------------------------------------------------------

    # Create cache folder to save features
    if not os.path.exists(PATH_CACHE):
        os.makedirs(PATH_CACHE)

    # Loop over all datasets
    for args.task in args.tasks:
        print("  Processing: [{dataset}]".format(dataset=args.task))

        # Identify task
        args.task = args.task

        # Get vlm id
        if args.vlm is None:  # Default vlm for each modality.
            args.vlm_id = task_to_vlm[args.task]
        else:
            args.vlm_id = args.vlm

        # Retrieve task details (i.e. experiment for ID, experiments for OOD, ...)
        get_task_setting(args)

        # Get pre-trained VLM setting
        get_model_config(args)

        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        args.setting = get_experiment_setting(args.task_setting["experiment"])

        # Get model
        if args.vlm_id == "conch":
            model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=args.model_config["weights_path"])
            model.to(device).float()
        elif args.vlm_id == "flair":
            model = VLMModel(vision_type=args.model_config["architecture"],
                             from_checkpoint=args.model_config["load_weights"],
                             weights_path=args.model_config["weights_path"],
                             vision_pretrained=args.model_config["init_imagenet"],
                             modality=args.setting["modality"],
                             vlm_id=args.model_config["vlm_id"]
                             )
            model.to(device).float().eval()
        elif args.vlm_id == "convirt":
            model = VLMModel(vision_type=args.model_config["architecture"],
                             from_checkpoint=args.model_config["load_weights"],
                             weights_path=args.model_config["weights_path"],
                             vision_pretrained=args.model_config["init_imagenet"],
                             modality=args.setting["modality"],
                             vlm_id=args.model_config["vlm_id"]
                             )
            model.to(device).float().eval()

        else:
            print("VLM not available....")
            return

        # Generate zero-shot prototypes
        initial_prototypes = get_text_prototypes(model, args.setting["targets"], vlm_id=args.vlm_id)

        # Set Adapter
        adapter = Adapter(initial_prototypes, model.logit_scale.item(), adapter="ZS")

        # Set training and test datasets
        datasets = [args.task_setting["experiment"]] + args.task_setting["experiment_test"]

        experiment = {"partitions": {}}
        # Set testing datasets
        for i in range(len(datasets)):
            experiment["partitions"][i] = {}
            experiment["partitions"][i]["domain"] = {datasets[i]}
            # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
            setting = get_experiment_setting(datasets[i])
            experiment["partitions"][i]["dataloader"] = set_loader(
                setting["dataframe"], args.data_root_path + setting["base_samples_path"],
                setting["targets"], size=args.model_config["size"], norm=args.model_config["norm"])

        # Test on different domains
        time_extraction = []
        for i_domain in range(0, len(experiment["partitions"])):
            print("  Processing: [{dataset}]".format(dataset=datasets[i_domain]))

            # Extract vision features
            time_adapt_i_1 = time.time()
            id = PATH_CACHE + datasets[i_domain] + "_" + args.vlm_id.lower().replace("/", "_")
            if not os.path.isfile(id + ".npz"):

                print("  Extracting features and saving in disk")
                if args.vlm_id == "conch":
                    feats_ds, refs_ds = extract_vision_features(
                        model.visual, experiment["partitions"][i_domain]["dataloader"])
                elif args.vlm_id == "flair":
                    feats_ds, refs_ds = extract_vision_features(
                        model.vision_model, experiment["partitions"][i_domain]["dataloader"])
                elif args.vlm_id == "convirt":
                    feats_ds, refs_ds = extract_vision_features(
                        model.vision_model, experiment["partitions"][i_domain]["dataloader"])

                print("  Extracting logits")
                logits_ds = predict_from_features(adapter, torch.tensor(feats_ds), bs=args.bs, act=False, epsilon=1.0)
                logits_ds = logits_ds.cpu().numpy()
                time_adapt_i_2 = time.time()

                print("  Saving in disk")
                np.savez(id, feats_ds=feats_ds, logits_ds=logits_ds, refs_ds=refs_ds,
                         logit_scale=model.logit_scale.item(), initial_prototypes=initial_prototypes.cpu().numpy())
            else:
                time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1
            time_extraction.append(time_adapt_i)
            print(str("Feature extraction time: " + str(datetime.timedelta(seconds=time_adapt_i))))
        print("Average time: " + str(datetime.timedelta(seconds=np.mean(time_extraction))))


def main():

    # Avoid multiple workers multiprocessing errors
    set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')

    # Tasks
    parser.add_argument('--tasks',
                        default='Gleason,Skin,NCT,MESSIDOR,MMAC,FIVES,CheXpert5x200,NIH,COVID',
                        help='Gleason,Skin,NCT,MESSIDOR,MMAC,FIVES,CheXpert5x200,NIH,COVID',
                        type=lambda s: [item for item in s.split(',')])

    # Pre-trained model to employ
    parser.add_argument('--vlm', default=None,
                        help='Pre-trained VLM to use (in case you want use a different than the pre-defined configs): '
                             '"conch ", "flair", "convirt"')

    # Other hyper-param
    parser.add_argument('--bs', default=128, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()