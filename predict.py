import os
import json
import argparse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

def load_and_prep_image(img_path):
    """Load and preprocess image for prediction."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict(image_path, model, idx_to_class):
    """Run prediction and return class label and probabilities."""
    img_array, display_img = load_and_prep_image(image_path)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = idx_to_class[pred_idx]
    return pred_class, preds[0], display_img

def visualize_results(pred_class, pred_prob, idx_to_class, display_img):
    """Visualize input image and prediction probabilities with value annotations."""
    classes = list(idx_to_class.values())
    probabilities = pred_prob * 100  # Convert to %

    plt.figure(figsize=(12, 6))

    # Show the input image
    plt.subplot(1, 2, 1)
    plt.imshow(display_img)
    plt.title(f"Predicted: {pred_class}", fontsize=14, color='green')
    plt.axis('off')

    # Plot the probability bar chart
    plt.subplot(1, 2, 2)
    bars = plt.bar(classes, probabilities)

    plt.xticks(rotation=45)
    plt.ylabel('Probability (%)')
    plt.title('Prediction Confidence (All Classes)')

    # Annotate each bar with its value
    for i, bar in enumerate(bars):
        prob_text = f"{probabilities[i]:.2f}%"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 prob_text, ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight predicted bar
        if classes[i] == pred_class:
            bar.set_color('green')
        else:
            bar.set_color('gray')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict kidney disease from CT image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/saved_model.h5', help='Path to saved model')
    parser.add_argument('--class_map', type=str, default='models/class_indices.json', help='Path to class indices json')
    args = parser.parse_args()

    print(f"Looking for image at: {args.image}")

    if not os.path.exists(args.image):
        print(f"❌ Error: Image path {args.image} does not exist.")
        return

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file {args.model} does not exist.")
        return

    if not os.path.exists(args.class_map):
        print(f"❌ Error: Class indices file {args.class_map} does not exist.")
        return

    print(f"✅ Loading model from: {args.model}")
    model = load_model(args.model)

    with open(args.class_map) as f:
        class_indices = json.load(f)

    # Reverse mapping: index -> class label
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Prediction
    pred_class, pred_prob, display_img = predict(args.image, model, idx_to_class)

    print(f"\n✅ Predicted Disease Type: {pred_class}")
    print("📊 Class Probabilities:")
    for i in range(len(pred_prob)):
        print(f"  {idx_to_class[i]}: {pred_prob[i] * 100:.2f}%")

    # Visualization
    visualize_results(pred_class, pred_prob, idx_to_class, display_img)

if __name__ == '__main__':
    main()
