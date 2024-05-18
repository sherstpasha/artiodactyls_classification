import argparse
import torch
from sklearn.metrics import f1_score
import timm
from functions import distribute_images_val_resnet

# Function to calculate F1 score
def calculate_f1_score(true_labels, pred_labels):
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=1)
    return f1

def main(input_folder, output_folder, model_path, device, class_names):
    # Load the pre-trained model from timm
    model = timm.create_model('resnext50_32x4d.a3_in1k', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Distribute images into folders and get labels
    true_labels, pred_labels, file_names, average_time_per_image = distribute_images_val_resnet(input_folder, output_folder, model, transforms, class_names, device)

    # Calculate F1 score
    f1 = calculate_f1_score(true_labels, pred_labels)
    print(f"F1 Score: {f1:.4f}")
    print(f"Average time per image: {average_time_per_image:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images and calculate F1 score.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder with images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save sorted images.")
    parser.add_argument("--model_path", type=str, default="best_model_ResNet_20.pth", help="Path to the saved model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on.")
    
    args = parser.parse_args()

    # Class names corresponding to the indices
    class_names = ["Кабарга", "Косуля", "Олень"]

    main(args.input_folder, args.output_folder, args.model_path, args.device, class_names)