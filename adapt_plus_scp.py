"""
Main function for conformal prediction using zero-shot, Conf-OT and baseline methods for transfer learning.
It includes three non-conformity scores: LAC, APS, and RAPS.
"""

import argparse
import torch
import os
import conformal

import numpy as np
import pandas as pd

from tqdm import tqdm

from conformal.metrics import evaluate_conformal, accuracy, aca
from datetime import datetime

from data.configs import get_task_setting, get_experiment_setting
from local_data.constants import *
from modeling.vlms.constants import *

from modeling.adapters.models import Adapter
from solvers import sstext

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # Prepare results storage
    results_detailed = {}
    res = pd.DataFrame()
    for i_task in args.tasks:
        print("  Testing on: [{dataset}]".format(dataset=i_task))

        # Identify task
        args.task = i_task

        # Get vlm id
        if args.vlm is None:  # Default vlm for each modality.
            args.vlm_id = task_to_vlm[args.task]
        else:
            args.vlm_id = args.vlm

        # Retrieve task details (i.e. experiment for ID, experiments for OOD, ...)
        get_task_setting(args)

        # ----------------------------------------
        # Load adaptation data
        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        setting = get_experiment_setting(args.task_setting["experiment"])
        print("  Adapting on: [{dataset}]".format(dataset=setting["experiment"]))
        # Load data
        id = "./local_data/cache/" + setting["experiment"] + "_" + args.vlm_id.lower().replace("/", "_")
        if os.path.isfile(id + ".npz"):
            print("  Loading features from cache_features")
            cache_adapt = np.load(id + ".npz", allow_pickle=True)
        else:
            print("Training data not found... return")
            return

        # ----------------------------------------
        # Load testing data
        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        setting = get_experiment_setting(args.task_setting["experiment_test"][0])
        print("  Adapting on: [{dataset}]".format(dataset=args.task_setting["experiment_test"][0]))
        # Load data
        id = "./local_data/cache/" + setting["experiment"] + "_" + args.vlm_id.lower().replace("/", "_")
        if os.path.isfile(id + ".npz"):
            print("  Loading features from cache_features")
            cache_test = np.load(id + ".npz", allow_pickle=True)
        else:
            print("Training data not found... return")
            return

        # Run for different seeds
        emp_cov, set_size, strat_covgap, class_covgap = [], [], [], []
        top1, bal_acu = [], []
        time_adapt, time_conf_fit, time_conf_inf = [], [], []

        for _ in tqdm(range(args.seeds), leave=False, desc="  Conformal inference: "):
            torch.cuda.empty_cache()

            # Get test features and labels
            feats_test, labels_test = cache_test["feats_ds"], np.int8(cache_test["refs_ds"])

            # Calibration partition
            if args.split_balanced:
                counts = np.ones((len(np.unique(labels_test))))
            else:  # Get label-marginal distribution from testing
                counts = np.bincount(labels_test)
            label_dist = counts / np.sum(counts)

            # Calibration split: retrieving few-shots from training partitions
            feats_calib, labels_calib = conformal.balance_split(
                cache_adapt["feats_ds"], np.int8(cache_adapt["refs_ds"]), k=args.k, p=label_dist, seed=_)

            # Move data to gpu
            feats_test, labels_test = torch.tensor(feats_test), torch.tensor(labels_test).to(torch.long)
            feats_calib, labels_calib = torch.tensor(feats_calib), torch.tensor(labels_calib).to(torch.long)

            # Set Adapter
            adapter = Adapter(torch.tensor(cache_adapt["initial_prototypes"]),
                              cache_adapt["logit_scale"], adapter=args.adapt)

            # Adjust Adapter based on calibration data
            preds_calib, adapter = sstext.adapt(feats_calib, labels_calib, adapter)

            # Predict on test
            with torch.no_grad():
                preds_test = torch.softmax(adapter(feats_test.to(device)), -1)

            # Set transfer learning times
            time_adapt_i = 0.0

            # apply the conformal algorithms
            val_sets, time_fit_i, time_infer_i = conformal.conformal_method(
                args.ncscore, preds_calib, labels_calib, preds_test, args.alpha)

            #  Run metrics
            metrics_conformal = evaluate_conformal(val_sets, labels_test, alpha=args.alpha)
            metrics_accuracy = accuracy(preds_test, labels_test, (1,))
            metrics_aca = aca(preds_test.cpu().numpy(), labels_test.cpu().numpy())
            # Allocate conformal inference metrics
            emp_cov.append(metrics_conformal[0]), set_size.append(metrics_conformal[1])
            class_covgap.append(metrics_conformal[2])
            # Training times
            time_adapt.append(time_adapt_i), time_conf_fit.append(time_fit_i), time_conf_inf.append(time_infer_i)
            # Allocate accuracy-related metrics
            top1.append(metrics_accuracy[0].item()), bal_acu.append(metrics_aca)

            # Output metrics
            print('  Empirical Coverage: [{cover}] -- Set Size: [{size}] -- '
                  'class_covgap: [{class_covgap}]'.format(cover=np.round(emp_cov[-1], 3),
                                                          size=np.round(set_size[-1], 2),
                                                          class_covgap=np.round(class_covgap[-1], 3)))
            print('  ACA: [{aca}]'.format(
                aca=np.round(np.median(bal_acu[-1]), 3)))

            # Save detailed results
        results_detailed[i_task] = {}
        results_detailed[i_task]["cov"] = emp_cov
        results_detailed[i_task]["set_size"] = set_size
        results_detailed[i_task]["class_covgap"] = class_covgap
        results_detailed[i_task]["top1"] = top1

        # Output metrics
        print("  " + "%" * 100)
        print('  [AVG] Empirical Coverage: [{cover}] -- Set Size: [{size}] -- '
              'class_covgap: [{class_covgap}]'.format(cover=np.round(np.median(emp_cov), 3),
                                                      size=np.round(np.median(set_size), 2),
                                                      class_covgap=np.round(np.median(class_covgap), 3)))
        print('  [AVG] ACA: [{aca}]'.format(
            aca=np.round(np.median(bal_acu), 3)))
        print("  " + "%" * 100)

        # Prepare results
        res_i = {"backbone": args.vlm_id, "dataset": args.task, "alpha": args.alpha,
                 "ncscore": args.ncscore, "shots": args.k, "split_balanced": str(args.split_balanced),
                 "top1": np.round(np.median(top1), 3), "aca": np.round(np.median(bal_acu), 3),
                 "cov": np.round(np.median(emp_cov), 3), "size": np.round(np.median(set_size), 2),
                 "CCV": np.round(np.median(class_covgap), 3), "time_adapt": np.round(np.mean(time_adapt), 6),
                 "time_conf_fit": np.round(np.mean(time_conf_fit), 6),
                 "time_conf_inf": np.round(np.mean(time_conf_inf), 6)}
        res = pd.concat([res, pd.DataFrame(res_i, index=[0])])

    # Produce average results
    avg = res[["top1", "aca", "cov", "size", "CCV", "time_adapt", "time_conf_fit", "time_conf_inf"]].mean().values
    res_avg = {"backbone": "AVG", "dataset": "AVG", "alpha": args.alpha,
               "ncscore": args.ncscore, "shots": args.k, "split_balanced": str(args.split_balanced),
               "top1": np.round(avg[0], 3), "aca": np.round(avg[1], 3),
               "cov": np.round(avg[2], 3), "size": np.round(avg[3], 2),
               "CCV": np.round(avg[4], 3), "time_adapt": np.round(avg[5], 6),
               "time_conf_fit": np.round(avg[6], 6), "time_conf_inf": np.round(avg[7], 6)}
    res = pd.concat([res, pd.DataFrame(res_avg, index=[0])])

    timestamp = datetime.now().strftime("-%m-%d_%H-%M-%S")
    # save summary results
    path = "./local_data/results/adapt_scp/{alpha}/{ncscore}/summary/".format(
        alpha=str(args.alpha).replace(".", ""), ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame.to_excel(res, path + args.adapt + timestamp + ".xlsx")

    # save detailed results
    path = "./local_data/results/adapt_scp/{alpha}/{ncscore}/detailed/".format(
        alpha=str(args.alpha).replace(".", ""), ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + args.adapt + timestamp + ".npy", results_detailed)


def main():
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

    # Setting for adaptation (OT hyper-parameters)
    parser.add_argument('--adapt', default='none', help='TL mode', choices=['none'])

    # Conformal prediction hyperparameters
    parser.add_argument('--alpha', default=0.10, help='Value for the desired coverage.', type=float)
    parser.add_argument('--ncscore', default='lac', help='Non-conformity score', choices=['lac', 'aps', 'raps'])

    # Experimental setting (data) hyperparameters
    parser.add_argument('--split_balanced', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--k', default=16, help='Number of shots for adaptation per class', type=int)

    # Number of seeds
    parser.add_argument('--seeds', default=20, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()