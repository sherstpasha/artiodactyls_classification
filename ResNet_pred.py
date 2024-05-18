import argparse
import torch
import timm
from functions import distribute_images_pred_resnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images and distribute them into folders.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder with images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save sorted images.")
    parser.add_argument("--model_path", type=str, default="best_model_ResNet_20.pth", help="Path to the saved model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on.")

    args = parser.parse_args()

    # Class names corresponding to the indices
    class_names = ["Кабарга", "Косуля", "Олень", "не найденно"]

    # Load the pre-trained model from timm
    model = timm.create_model('resnext50_32x4d.a3_in1k', pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Distribute images into folders and get labels
    distribute_images_pred_resnet(args.input_folder, args.output_folder, model, transforms, class_names, args.device)