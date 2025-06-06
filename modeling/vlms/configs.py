
def get_model_config(args):

    if args.vlm_id == "convirt":
        model_config = {"vlm_id": args.vlm_id, "weights_path": "./local_data/models_weights/cxr_clip_resnet.pth",
                        "load_weights": True, "init_imagenet": True, "architecture": "resnet", "size": 224,
                        "norm": False}
    elif args.vlm_id == "flair":
        model_config = {"vlm_id": args.vlm_id, "weights_path": "./local_data/models_weights/flair_resnet.pth",
                        "load_weights": True, "init_imagenet": True, "architecture": "resnet", "size": 512,
                        "norm": False}
    elif args.vlm_id == "conch":
        model_config = {"vlm_id": args.vlm_id, "weights_path": "./local_data/models_weights/conch.bin",
                        "load_weights": True, "init_imagenet": True, "architecture": "vitb16", "size": 448,
                        "norm": True}
    else:
        model_config = None

    # Parse keys into argument parser
    args.model_config = model_config

