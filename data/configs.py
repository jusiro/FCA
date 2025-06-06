
def get_task_setting(args):

    if args.task == "Gleason":
        task_setting = {"experiment": "SICAPv2", "experiment_test": ["SICAPv2_test"]}
    elif args.task == "MITOSIS":
        task_setting = {"experiment": "MIDOG_A_train", "experiment_test": ["MIDOG_A_test"]}
    elif args.task == "Skin":
        task_setting = {"experiment": "Skin_train", "experiment_test": ["Skin_test"]}
    elif args.task == "NCT":
        task_setting = {"experiment": "NCT_train", "experiment_test": ["NCT_test"]}
    elif args.task == "MESSIDOR":
        task_setting = {"experiment": "MESSIDOR_train", "experiment_test": ["MESSIDOR_test"]}
    elif args.task == "MMAC":
        task_setting = {"experiment": "MMAC_A_train", "experiment_test": ["MMAC_A_test"]}
    elif args.task == "FIVES":
        task_setting = {"experiment": "FIVES_train", "experiment_test": ["FIVES_test"]}
    elif args.task == "CheXpert5x200":
        task_setting = {"experiment": "CheXpert5x200_train", "experiment_test": ["CheXpert5x200_test"]}
    elif args.task == "NIH":
        task_setting = {"experiment": "nihlt_train", "experiment_test": ["nihlt_test"]}
    elif args.task == "COVID":
        task_setting = {"experiment": "covid_train", "experiment_test": ["covid_test"]}
    else:
        print("Task not implemented... using CX5x200")
        task_setting = {"experiment": "SICAPv2", "experiment_test": ["SICAPv2_test"]}

    # Parse keys into argparser
    args.task_setting = task_setting


def get_experiment_setting(experiment: object) -> object:

    # Histology experiments
    if experiment == "SICAPv2":
        setting = {"experiment": "SICAPv2",
                   "targets": ["NC", "G3", "G4", "G5"],
                   "dataframe": "./local_data/dataframes/histology/SICAPv2_train.csv",
                   "base_samples_path": "HISTOLOGY/SICAPv2/images/",
                   "modality": "histology"
                   }
    elif experiment == "SICAPv2_test":
        setting = {"experiment": "SICAPv2_test",
                   "targets": ["NC", "G3", "G4", "G5"],
                   "dataframe": "./local_data/dataframes/histology/SICAPv2_test.csv",
                   "base_samples_path": "HISTOLOGY/SICAPv2/images/",
                   "modality": "histology"
                   }
    elif experiment == "NCT_train":
        setting = {"experiment": "NCT_train",
                   "targets": ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                   "Normal colon mucosa", "Cancer-associated stroma",
                   "Colorectal adenocarcinoma epithelium"],
                   "dataframe": "./local_data/dataframes/histology/NCTCRC_train.csv",
                   "base_samples_path": "HISTOLOGY/NCT-CRC/",
                   "modality": "histology"
                   }
    elif experiment == "NCT_test":
        setting = {"experiment": "NCT_test",
                   "targets": ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                   "Normal colon mucosa", "Cancer-associated stroma",
                   "Colorectal adenocarcinoma epithelium"],
                   "dataframe": "./local_data/dataframes/histology/NCTCRC_test.csv",
                   "base_samples_path": "HISTOLOGY/NCT-CRC/",
                   "modality": "histology"
                   }
    elif experiment == "Skin_train":
        setting = {"experiment": "Skin_train",
                   "targets": ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands', 'Vessels', 'Elastosis',
                               'Chondral tissue', 'Hair follicle', 'Epidermis', 'Nerves', 'Subcutis', 'Dermis',
                               'Sebaceous glands', 'Squamous-cell carcinoma', 'Melanoma in-situ', 'Basal-cell carcinoma',
                               'Naevus'],
                   "dataframe": "./local_data/dataframes/histology/Skin_train.csv",
                   "base_samples_path": "HISTOLOGY/Skin/",
                   "modality": "histology"
                   }
    elif experiment == "Skin_test":
        setting = {"experiment": "Skin_test",
                   "targets": ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands', 'Vessels', 'Elastosis',
                               'Chondral tissue', 'Hair follicle', 'Epidermis', 'Nerves', 'Subcutis', 'Dermis',
                               'Sebaceous glands', 'Squamous-cell carcinoma', 'Melanoma in-situ', 'Basal-cell carcinoma',
                               'Naevus'],
                   "dataframe": "./local_data/dataframes/histology/Skin_test.csv",
                   "base_samples_path": "HISTOLOGY/Skin/",
                   "modality": "histology"
                   }

    # Ophthalmology experiments
    elif experiment == "MESSIDOR_train":
        setting = {"experiment": "MESSIDOR_train",
                   "targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "./local_data/dataframes/fundus/MESSIDOR_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/02_MESSIDOR/",
                   "modality": "fundus"
                   }
    elif experiment == "MESSIDOR_test":
        setting = {"experiment": "MESSIDOR_test",
                   "targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "./local_data/dataframes/fundus/MESSIDOR_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/02_MESSIDOR/",
                   "modality": "fundus"
                   }
    elif experiment == "MMAC_A_train":
        setting = {"experiment": "MMAC_A_train",
                   "targets": ["no retinal lesion", "tessellated fundus", "diffuse chorioretinal atrophy",
                               "patchy chorioretinal atrophy", "macular atrophy"],
                   "dataframe": "./local_data/dataframes/fundus/MMAC_A_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/38_MMAC23/"
                                        "1.Classification/1.Images/1.Training/",
                   "modality": "fundus"
                   }
    elif experiment == "MMAC_A_test":
        setting = {"experiment": "MMAC_A_test",
                   "targets": ["no retinal lesion", "tessellated fundus", "diffuse chorioretinal atrophy",
                               "patchy chorioretinal atrophy", "macular atrophy"],
                   "dataframe": "./local_data/dataframes/fundus/MMAC_A_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/38_MMAC23/"
                                        "1.Classification/1.Images/2.Validation/",
                   "modality": "fundus"
                   }
    elif experiment == "FIVES_train":
        setting = {"experiment": "FIVES_train",
                   "targets": ['normal', 'diabetic retinopathy', 'glaucoma', 'age related macular degeneration'],
                   "dataframe": "./local_data/dataframes/fundus/FIVES_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/",
                   "modality": "fundus"
                   }
    elif experiment == "FIVES_test":
        setting = {"experiment": "FIVES_test",
                   "targets": ['normal', 'diabetic retinopathy', 'glaucoma', 'age related macular degeneration'],
                   "dataframe": "./local_data/dataframes/fundus/FIVES_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/",
                   "modality": "fundus"
                   }

    # Radiology experiments
    elif experiment == "CheXpert5x200_train":
        setting = {"experiment": "CheXpert5x200_train",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/dataframes/cxr/CheXpert5x200_train.csv",
                   "base_samples_path": "CXR/CheXpert/CheXpert-v1.0/",
                   "modality": "cxr"
        }
    elif experiment == "CheXpert5x200_test":
        setting = {"experiment": "CheXpert5x200_test",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/dataframes/cxr/CheXpert5x200_test.csv",
                   "base_samples_path": "CXR/CheXpert/CheXpert-v1.0/",
                   "modality": "cxr"
        }
    elif experiment == "nihlt_train":
        setting = {"experiment": "nihlt_train",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/dataframes/cxr/nih_train.csv",
                   "base_samples_path": "CXR/NIH/",
                   "modality": "cxr"
                   }
    elif experiment == "nihlt_test":
        setting = {"experiment": "nihlt_test",
                   "targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "./local_data/dataframes/cxr/nih_test.csv",
                   "base_samples_path": "CXR/NIH/",
                   "modality": "cxr"
                   }
    elif experiment == "covid_train":
        setting = {"experiment": "covid_train",
                   "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "./local_data/dataframes/cxr/covid_train.csv",
                   "base_samples_path": "CXR/COVID-19_Radiography_Dataset/",
                   "modality": "cxr"
                   }
    elif experiment == "covid_test":
        setting = {"experiment": "covid_test",
                   "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "./local_data/dataframes/cxr/covid_test.csv",
                   "base_samples_path": "CXR/COVID-19_Radiography_Dataset/",
                   "modality": "cxr"
                   }
    else:
        setting = None
        print("Experiment not prepared...")

    return setting
