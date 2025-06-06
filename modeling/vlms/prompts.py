
import random

CATEGORIES_CXR = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Normal", "Pneumothorax"]

CATEGORIES_ALL_CXR = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
                  "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                  "Pleural Other", "Fracture", "Support Devices", "Normal", "COVID", "Infiltration", "Mass",
                  "Nodule", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum", "Pneumomediastinum",
                  "Subcutaneous Emphysema", "Tortuous Aorta", "Calcification of the Aorta", "Bronchitis",
                  "Brocho-pneumonia", "Bronchiolitis", "Situs Inversus", "Pleuropneumonia", "Diafragmatic hernia",
                  "Tuberculosis", "Congenital Pulmonary Airwat Malformation", "Hyaline Membrane Disease",
                  "Mediastinal Tumor", "Lung Tumor", "Effusion"]


ASSEMBLE_PROMPTS_CXR = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": ["subsegmental atelectasis", "linear atelectasis", "trace atelectasis", "bibasilar atelectasis",
                    "retrocardiac atelectasis", "bandlike atelectasis", "residual atelectasis"],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe"],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": ["cardiac silhouette size is upper limits of normal", "cardiomegaly which is unchanged",
                    "mildly prominent cardiac silhouette", "portable view of the chest demonstrates stable cardiomegaly",
                    "portable view of the chest demonstrates mild cardiomegaly", "persistent severe cardiomegaly",
                    "heart size is borderline enlarged", "cardiomegaly unchanged",
                    "heart size is at the upper limits of normal", "redemonstration of cardiomegaly",
                    "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
                    "cardiac silhouette size is mildly enlarged",
                    "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
                    "heart size remains at mildly enlarged",
                    "persistent cardiomegaly with prominent upper lobe vessels"],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": ["bilateral consolidation", "reticular consolidation", "retrocardiac consolidation",
                    "patchy consolidation", "airspace consolidation", "partial consolidation"],
        "location": ["at the lower lung zone", "at the upper lung zone", "at the left lower lobe",
                     "at the right lower lobe", "at the left upper lobe", "at the right uppper lobe",
                     "at the right lung base", "at the left lung base"],
    },
    "Edema": {
        "severity": ["", "mild", "improvement in", "presistent", "moderate", "decreased"],
        "subtype": ["edema", "pulmonary edema", "trace interstitial edema", "pulmonary interstitial edema"],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": ["pleural effusion", "bilateral pleural effusion", "subpulmonic pleural effusion",
                    "bilateral pleural effusion"],
    },
    "Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": ["pleural effusion", "bilateral effusion", "subpulmonic effusion",
                    "bilateral effusion"],
    },
    "Pneumothorax": {
        "severity": [""],
        "subtype": ["pneumothorax"],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe",
                     "at the left middle lobe", "at the right middle lobe"],
    },
    "Pneumonia": {
        "severity": ['round', 'early', 'focal', 'multifocal', 'small', ''],
        "subtype": ["pneumonia", 'bacterial', 'viral', 'mycoplasma', ''],
        "location": ["at the mid lung zone", "at the upper lung zone", "at the right lung zone",
                     "at the left lung zone", "at the lung bases", "at the right lung base", "at the left lung base",
                     "at the bilateral lung bases", "at the left lower lobe", "at the right lower lobe",
                     "at the left middle lobe", "at the right middle lobe"],
    },
    "COVID": {
        "severity": ["patchy", "confluent"],
        "description": ["ground glass"],
        "subtype": ["covid", "opacity", "consolidation"],
        "location": ["in peripheral", "in mid", "in lower"],
    },
    "Normal": {
        "severity": [""],
        "description": [""],
        "subtype": ["normal", "no findings"],
        "location": [""],
    },
}

DESCRIPTIONS_PROMPTS = {"No Finding": ["no findings"],
                        "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum"],
                        "Cardiomegaly": ["the heart is enlarged", "cardiomegaly"],
                        "Lung Lesion": ["lung lesion"],
                        "Lung Opacity": ["area of hazy opacification due to air displacement by fluid, airway collapse,"
                                         " fibrosis, or a neoplastic process." "It is causes include infections,"
                                         " interstitial lung disease, and pulmonary edema", "lung opacity"],
                        "Edema": ["pulmonary congestion", "excessive liquid accumulation in the tissue and air spaces"
                                  " of the lungs", "fluid in the alveolar walls", "edema"],
                        "Consolidation": ["region of normally compressible lung tissue that has filled with "
                                          " instead of air", "consolidation"],
                        "Pneumonia": ["pneumonia is an inflammatory condition of the lung primarily small air sacs"
                                      " known as alveoli", "pneumonia may present with opacities", "Complications such"
                                      " as pleural effusion may also be found increasing the diagnostic accuracy of "
                                      "lung consolidation and pleural effusion", "pneumonia"],
                        "Atelectasis": ["collapse or closure of a lung resulting in reduced or absent gas exchange",
                                        "Findings can include lung opacification and loss of lung volume",
                                        "atelectasis"],
                        "Pneumothorax": ["abnormal collection of air in the pleural space between the lung and the "
                                         "chest wall", "it may be caused by pneumonia or fibrosis and other diseases",
                                         "pneumothorax"],
                        "Pleural Effusion": ["pleural Effusion"],
                        "Pleural Other": ["pleural lesion"],
                        "Fracture": ["break in a rib bone", "fracture"],
                        "Support Devices": ["support devices"],
                        "Normal": ["absence of diseases and infirmity findings, indicating the structure is normal.",
                                   "no findings"],
                        "COVID": ["it is a contagious disease caused by a virus.", "ground-glass opacities,"
                                  " consolidation, thickening, pleural effusions commonly appear in infection."],
                        "Infiltration": ["infiltration", "substance denser than air, such as pus, blood, or protein,"
                                         " which lingers within the parenchyma of the lungs."],
                        "Mass": ["mass"],
                        "Nodule": ["nodule"],
                        "Emphysema": ["emphysema", "lower respiratory tract disease, characterized by air-filled spaces"
                                      "in the lungs, that can vary in size and may be very large."],
                        "Fibrosis": ["fibrosis"],
                        "Pleural Thickening": ["pleural thickening"],
                        "Pneumoperitoneum": ["pneumoperitoneum"],
                        "Pneumomediastinum": ["pneumomediastinum"],
                        "Subcutaneous Emphysema": ["subcutaneous emphysema"],
                        "Tortuous Aorta": ["tortuous aorta", "aorta is slightly tortuous", "varicose veins"],
                        "Calcification of the Aorta": ["calcification of the aorta"],
                        "Bronchitis": ["bronchitis"],
                        "Broncho-pneumonia": ["broncho-pneumonia"],
                        "Bronchiolitis": ["bronchiolitis"],
                        "Situs Inversus": ["situs inversus"],
                        "Pleuropneumonia": ["pleuropneumonia"],
                        "Diafragmatic hernia": ["diafragmatic hernia"],
                        "Tuberculosis": ["tuberculosis"],
                        "Congenital Pulmonary Airwat Malformation": ["congenital pulmonary airwat malformation"],
                        "Hyaline Membrane Disease": ["hyaline membrane disease"],
                        "Mediastinal Tumor": ["mediastinal tumor"],
                        "Lung Tumor": ["lung tumor"],
                        "Effusion": ["pleural Effusion"],
                        }


def generate_prompt_cxr(n=100):

    # Generate missing categories
    for iCategory in CATEGORIES_ALL_CXR:
        if iCategory not in list(ASSEMBLE_PROMPTS_CXR.keys()):
            ASSEMBLE_PROMPTS_CXR[iCategory] = {}
            ASSEMBLE_PROMPTS_CXR[iCategory]["severity"] = {""}
            if iCategory in list(DESCRIPTIONS_PROMPTS.keys()):
                ASSEMBLE_PROMPTS_CXR[iCategory]["description"] = DESCRIPTIONS_PROMPTS[iCategory]
            else:
                ASSEMBLE_PROMPTS_CXR[iCategory]["description"] = {iCategory}
            ASSEMBLE_PROMPTS_CXR[iCategory]["subtype"] = {iCategory}
            ASSEMBLE_PROMPTS_CXR[iCategory]["location"] = {""}

    # Generate prompts
    prompts = {}
    for k, v in ASSEMBLE_PROMPTS_CXR.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    prompt = f"{k0} {k1} {k2}".replace("   ", " ").replace("  ", " ")
                    if prompt[0] == " ":
                        prompt = prompt[1:]
                    if prompt[-1] == " ":
                        prompt = prompt[:-1]
                    cls_prompts.append(prompt.lower())

        # randomly sample n prompts for zero-shot classification
        if n is not None and n < int(len(cls_prompts)):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
    return prompts


def generate_prompt_fundus(categories):
    caption = "A fundus photograph of [CLS]"
    prompts_dict = {"no diabetic retinopathy": ["no diabetic retinopathy", "no microaneurysms"],
                    "mild diabetic retinopathy": ["only few microaneurysms"],
                    "moderate diabetic retinopathy": ["many exudates near the macula", "hard exudates",
                                                      "many haemorrhages near the macula", "few severe haemorrhages",
                                                      "retinal thickening near the macula", "cotton wool spots"],
                    "severe diabetic retinopathy": ["venous beading", "many severe haemorrhages",
                                                    "intraretinal microvascular abnormality"],
                    "proliferative diabetic retinopathy": ["preretinal or vitreous haemorrhage", "neovascularization"],
                    "no retinal lesion": ["no retinal lesion"],
                    "tessellated fundus": ["tessellated fundus"],
                    "diffuse chorioretinal atrophy": ["diffuse chorioretinal atrophy"],
                    "patchy chorioretinal atrophy": ["patchy chorioretinal atrophy"],
                    "macular atrophy": ["macular atrophy"],
                    "noHR": ['no presence of hypertensive retinopathy', 'normal', 'no findings', 'normal optic disk'],
                    "HR": ['possible signs of haemorraghe with blot, dot, or flame-shaped',
                           'possible presence of microaneurysm, cotton-wool spot, or hard exudate',
                           'arteriolar narrowing', 'vascular wall changes', 'optic disk edema'],
                    "NG": ["no glaucoma"],
                    "G":  ["optic nerve abnormalities", "abnormal size of the optic cup",
                           "anomalous size in the optic disc"],
                    "normal": ["healthy", "no findings", "no lesion signs", "no glaucoma", "no retinopathy"],
                    "age related macular degeneration": ["many small drusen", "few medium-sized drusen", "large drusen",
                                                         "macular degeneration"],
                    "diabetic retinopathy": ["diabetic retinopathy"],
                    "glaucoma": ["optic nerve abnormalities", "abnormal size of the optic cup",
                                 "anomalous size in the optic disc"],
                    }

    prompts = {}
    for iCategory in categories:
        prompts[iCategory] = [caption.replace("[CLS]", iDescription) for iDescription in prompts_dict[iCategory]]
    return prompts


def generate_prompt_histology(categories, model_id="conch"):

    if "conch" in model_id:
        captions = ["[CLS].",
                    "a photomicrograph showing [CLS].",
                    "a photomicrograph of [CLS].",
                    "an image of [CLS].",
                    "an image showing [CLS].",
                    "an example of [CLS].",
                    "[CLS] is shown.",
                    "this is [CLS].",
                    "there is [CLS].",
                    "a histopathological image showing [CLS].",
                    "a histopathological image of [CLS].",
                    "a histopathological photograph of [CLS].",
                    "a histopathological photograph showing [CLS].",
                    "shows [CLS].",
                    "presence of [CLS].",
                    "[CLS] is present.",
                    "an H&E stained image of [CLS].",
                    "an H&E stained image showing [CLS].",
                    "an H&E image showing [CLS].",
                    "an H&E image of [CLS].",
                    "[CLS], H&E stain.",
                    "[CLS], H&E."]

        prompts_dict = {"NC": ["non-cancerous tissue", "non-cancerous prostate tissue", "benign tissue",
                               "benign glands", "benign prostate glands", "benign prostate tissue"],
                        "G3": ["gleason grade 3", "gleason pattern 3", "prostate cancer, gleason grade 3",
                               "prostate cancer, gleason pattern 3", "prostate adenocarcinoma, well-differentiated",
                               "well-differentiated prostatic adenocarcinoma"],
                        "G4": ["gleason grade 4", "gleason pattern 4", "prostate cancer, gleason grade 4",
                               "prostate cancer, gleason pattern 4", "prostate adenocarcinoma, moderately differentiated",
                               "moderately differentiated prostatic adenocarcinoma"],
                        "G5": ["gleason grade 5", "gleason pattern 5", "prostate cancer, gleason grade 5",
                               "prostate cancer, gleason pattern 5", "prostate adenocarcinoma, poorly differentiated",
                               "poorly differentiated prostatic adenocarcinoma"],
                        "NT": ["lymph node"],
                        "T": ["lymph node containing metastatic tumor tissue"],
                        "NM": ["normal cells", "no mitosis"],
                        "M": ["mitosis", "atypical mitosis", "atypical cells"],
                        'Necrosis': ['necrosis'],
                        'Skeletal muscle': ['skeletal muscle'],
                        'Eccrine sweat glands': ['eccrine sweat glands'],
                        'Vessels': ['vessels'],
                        'Elastosis': ['elastosis'],
                        'Chondral tissue': ['chondral tissue'],
                        'Hair follicle': ['hair follicle'],
                        'Epidermis': ['epidermis'],
                        'Nerves': ['nerves'],
                        'Subcutis': ['subcutis'],
                        'Dermis': ['dermis'],
                        'Sebaceous glands': ['sebaceous glands'],
                        'Squamous-cell carcinoma': ['squamous-cell carcinoma'],
                        'Melanoma in-situ': ['melanoma in-situ'],
                        'Basal-cell carcinoma': ['basal-cell carcinoma'],
                        'Naevus': ['naevus'],
                        'Adipose': ['adipose'],
                        'Background': ['background'],
                        'Debris': ['debris'],
                        'Lymphocytes': ['lymphocytes'],
                        'Mucus': ['mucus'],
                        'Smooth muscle': ['smooth muscle'],
                        'Normal colon mucosa': ['normal colon mucosa'],
                        "Cancer-associated stroma": ["cancer-associated stroma"],
                        'Colorectal adenocarcinoma epithelium': ['colorectal adenocarcinoma epithelium'],
                        }

    else:
        captions = ["a histopathology slide showing [CLS]", "histopathology image of [CLS]",
                    "pathology tissue showing [CLS]", "presence of [CLS] tissue on image"]

        prompts_dict = {"NC": ["benign glands"],
                        "G3": ["atrophic dense glands"],
                        "G4": ["cribriform ill-formed fused papillary patterns"],
                        "G5": ["isolated nest cells without lumen roseting patterns"],
                        "NT": ["lymph node"],
                        "T":  ["lymph node containing metastatic tumor tissue"],
                        "NM": ["normal cells", "no mitosis"],
                        "M": ["mitosis", "atypical mitosis", "atypical cells"]
                        }

    prompts = {}
    for iCategory in categories:
        prompts[iCategory] = [caption.replace("[CLS]", iDescription) for iDescription in prompts_dict[iCategory]
                              for caption in captions]
    return prompts
