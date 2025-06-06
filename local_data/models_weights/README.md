We employ pre-trained vision-language model for ophthalmology, radiology, and histology. You can find the model weights
in the following links, and the expected folder structure is as below:

- Histology: [CONCH](https://huggingface.co/MahmoodLab/CONCH).
- Ophthalmology, fundus color images: [FLAIR](https://github.com/jusiro/FLAIR).
- Radiology, chest X-ray: [CONVIRT](https://drive.google.com/file/d/1YUFjy_lNZTuUMqcLcaKFjBPOt7hnpCUs/view?usp=sharing), pre-trained on MIMIC dataset.

```
FCA/
└── local_data/
    └── model_weights/
        ├── conch.bin
        ├── cxr_clip_resnet.pth
        └── flair_resnet.pth
```