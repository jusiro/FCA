import pandas as pd
import numpy as np
import torch
import os
import random
import conformal

from local_data.constants import PATH_DATASETS


def create_chexpert_dataframes():

    # Labels
    cat_to_num_map = {'Atelectasis': 0,
                      'Cardiomegaly': 1,
                      'Consolidation': 2,
                      'Edema': 3,
                      'Pleural Effusion': 4,
                      }

    # Labels names
    class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

    # Read dataframe
    df = pd.read_csv("./local_data/dataframes/cxr/chexpert_5x200.csv")

    # Create labels
    df = df[["Path"] + class_names]

    # Create train/test partition
    idx = np.arange(0, len(df))
    labels = np.argmax(df[class_names].values, axis=-1)

    # Calibration split: retrieving few-shots from training partitions
    idx_train, labels_train, idx_test, labels_test = conformal.standard_split(idx, labels, p=0.8)

    # Get df train
    df_train = df[[i in list(idx_train) for i in idx]]
    df_train.reset_index()

    # Get df test
    df_test = df[[i in list(idx_test) for i in idx]]
    df_test.reset_index()

    # Save dataframe
    df_train.to_csv("./local_data/dataframes/cxr/CheXpert5x200_train.csv")
    df_test.to_csv("./local_data/dataframes/cxr/CheXpert5x200_test.csv")


def create_fives_dataframes():

    # Labels
    cat_to_num_map = {'normal': 0,
                      'diabetic retinopathy': 1,
                      'glaucoma': 2,
                      'age related macular degeneration': 3,
                      }

    # Labels names
    class_names = ['normal', 'diabetic retinopathy', 'glaucoma',
                   'age related macular degeneration']

    # Read dataframe
    df = pd.read_csv("./local_data/dataframes/fundus/13_FIVES.csv")

    # Create Path column
    df["Path"] = df["image"]

    # Create labels
    all_cats = [eval(df["categories"][i])[0] for i in range(len(df))]
    df["class"] = all_cats

    # Map labels to one-hot encoding
    labels_cat = np.array(df["class"].map(cat_to_num_map))
    labels_cat_ohe = torch.nn.functional.one_hot(torch.tensor(labels_cat)).numpy()

    # Labels dataframe
    labels_df = pd.DataFrame(data=labels_cat_ohe, columns=class_names)

    # Join dataframe
    df_joint = pd.concat([df, labels_df], axis=1)

    # Clean columns
    df = df_joint[["Path"] + class_names]

    # Create train/test partition
    idx = np.arange(0, len(df))
    labels = np.argmax(df[class_names].values, axis=-1)

    # Calibration split: retrieving few-shots from training partitions
    idx_train, labels_train, idx_test, labels_test = conformal.standard_split(idx, labels, p=0.8)

    # Get df train
    df_train = df[[i in list(idx_train) for i in idx]]
    df_train.reset_index()

    # Get df test
    df_test = df[[i in list(idx_test) for i in idx]]
    df_test.reset_index()

    # Save dataframe
    df_train.to_csv("./local_data/dataframes/fundus/FIVES_train.csv")
    df_test.to_csv("./local_data/dataframes/fundus/FIVES_test.csv")


def create_mesidor_dataframes():

    # Set labels names
    labels_names = ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                    "severe diabetic retinopathy", "proliferative diabetic retinopathy"]

    # Read dataframe
    df = pd.read_csv("./local_data/dataframes/fundus/MESSIDOR.csv")

    # Create train/test partition
    idx = np.arange(0, len(df))
    labels = np.argmax(df[labels_names].values, axis=-1)

    # Calibration split: retrieving few-shots from training partitions
    idx_train, labels_train, idx_test, labels_test = conformal.standard_split(idx, labels, p=0.8)

    # Get df train
    df_train = df[[i in list(idx_train) for i in idx]]
    df_train.reset_index()
    
    # Get df test
    df_test = df[[i in list(idx_test) for i in idx]]
    df_test.reset_index()
    
    # Save dataframe
    df_train.to_csv("./local_data/dataframes/fundus/MESSIDOR_train.csv")
    df_test.to_csv("./local_data/dataframes/fundus/MESSIDOR_test.csv")


def create_skin_dataframes():

    # Labels
    cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                      'nontumor_skin_muscle_skeletal': 1,
                      'nontumor_skin_sweatglands_sweatglands': 2,
                      'nontumor_skin_vessel_vessel': 3,
                      'nontumor_skin_elastosis_elastosis': 4,
                      'nontumor_skin_chondraltissue_chondraltissue': 5,
                      'nontumor_skin_hairfollicle_hairfollicle': 6,
                      'nontumor_skin_epidermis_epidermis': 7,
                      'nontumor_skin_nerves_nerves': 8,
                      'nontumor_skin_subcutis_subcutis': 9,
                      'nontumor_skin_dermis_dermis': 10,
                      'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                      'tumor_skin_epithelial_sqcc': 12,
                      'tumor_skin_melanoma_melanoma': 13,
                      'tumor_skin_epithelial_bcc': 14,
                      'tumor_skin_naevus_naevus': 15
                      }

    # Labels names
    class_names = ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands',
                   'Vessels', 'Elastosis', 'Chondral tissue', 'Hair follicle',
                   'Epidermis', 'Nerves', 'Subcutis', 'Dermis', 'Sebaceous glands',
                   'Squamous-cell carcinoma', 'Melanoma in-situ',
                   'Basal-cell carcinoma', 'Naevus']

    # Read dataframe
    df = pd.read_csv(PATH_DATASETS + "HISTOLOGY/Skin/data/tiles-v2.csv")

    # Map labels to one-hot encoding
    labels_cat = np.array(df["class"].map(cat_to_num_map))
    labels_cat_ohe = torch.nn.functional.one_hot(torch.tensor(labels_cat)).numpy()

    # Labels dataframe
    labels_df = pd.DataFrame(data=labels_cat_ohe, columns=class_names)

    # Join dataframe
    df_joint = pd.concat([df, labels_df], axis=1)

    # Get path column
    df_joint["Path"] = df_joint["file"]

    # Partition
    data_train = df_joint[df_joint['set'] == 'Train']
    data_test = df_joint[df_joint['set'] == 'Test']

    # Retrieve columns of interest
    data_train = data_train[["Path"]+class_names]
    data_train = data_train.reset_index(drop=True)
    data_test = data_test[["Path"]+class_names]
    data_test = data_test.reset_index(drop=True)

    # Save dataframe
    data_train.to_csv("./local_data/dataframes/histology/Skin_train.csv")
    data_test.to_csv("./local_data/dataframes/histology/Skin_test.csv")


def create_ntccrc_dataframes():

    def read_data(path_base, path_dataset, subset):
        image_dir = path_base + path_dataset + subset
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        paths, labels = [], []

        data_count = 0
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            for imname in imnames:
                impath = os.path.join(os.path.join(subset, folder), imname)
                paths.append(impath)
                labels.append(label)
                data_count += 1
        print(data_count)
        return paths, labels


    BASE_PATH = "HISTOLOGY/NCT-CRC/"

    TO_BE_IGNORED = ["README.txt"]

    class_names = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                   "Normal colon mucosa", "Cancer-associated stroma",
                   "Colorectal adenocarcinoma epithelium"]

    # Read train and test data paths
    train_paths, train_labels = read_data(PATH_DATASETS, BASE_PATH, "NCT-CRC-HE-100K")
    test_paths, test_labels = read_data(PATH_DATASETS, BASE_PATH, "CRC-VAL-HE-7K")

    # Create Path dataframe
    df_train = pd.DataFrame(data=train_paths, columns=["Path"])
    df_test = pd.DataFrame(data=test_paths, columns=["Path"])

    # Create labels dataframe
    labels_cat_ohe_train = torch.nn.functional.one_hot(torch.tensor(train_labels)).numpy()
    labels_df_train = pd.DataFrame(data=labels_cat_ohe_train, columns=class_names)
    labels_cat_ohe_test = torch.nn.functional.one_hot(torch.tensor(test_labels)).numpy()
    labels_df_test = pd.DataFrame(data=labels_cat_ohe_test, columns=class_names)

    # Create joint dataframes and save
    df_train = pd.concat([df_train, labels_df_train], axis=1)
    df_test = pd.concat([df_test, labels_df_test], axis=1)

    # Save dataframe
    df_train.to_csv("./local_data/dataframes/histology/NCTCRC_train.csv")
    df_test.to_csv("./local_data/dataframes/histology/NCTCRC_test.csv")


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items