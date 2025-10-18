import torch
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import os

# Load CLIP ViT-B-32 (OpenAI weights) and preprocessor/tokenizer
clip_model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
device = "cpu"  # Use "cuda" if you have a GPU

# labels = [
#     "graph", "bar chart", "pie chart", "line chart", "table", "map", "plot", 
#     "statistical diagram", "comparison", "numeric data", "scatterplot", "data visualisation"
# ]

# text_tokens = tokenizer(labels).to(device)

# def is_relevant_image(img_path, min_prob=0.30):
#     """
#     Uses CLIP to check if image is a relevant visual/statistical type.
#     Returns True if match, otherwise False.
#     """
#     try:
#         img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
#         with torch.no_grad():
#             img_features = clip_model.encode_image(img)
#             text_features = clip_model.encode_text(text_tokens)
#             logits = (img_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
#             max_idx = logits.argmax()
#             max_label = labels[max_idx]
#             max_score = logits[max_idx]
#         return max_label in labels and max_score > min_prob
#     except Exception as e:
#         print(f"CLIP filtering error for {img_path}: {e}")
#         return False

accept_labels = [
    "graph", "bar chart", "pie chart", "line chart", "table", "map", "plot", 
    "statistical diagram", "comparison", "numeric data", "scatterplot", "data visualisation",
    "slide", "infographic", "presentation slide", "business presentation", "statistical slide",
    "flow chart", "dashboard", "heatmap", "summary table", "timeline"
]

reject_labels = [
    "human", "person", "people", "face", "selfie", "crowd", "portrait", "student", "teacher", "group photo"
]

all_labels = accept_labels + reject_labels

text_tokens = tokenizer(all_labels).to(device)

def is_relevant_image(img_path, min_prob=0.30):
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text_tokens)
        logits = (img_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
        max_idx = logits.argmax()
        max_label = all_labels[max_idx]
        max_score = logits[max_idx]
    # Accept image only if its label is in accept_labels AND NOT in reject_labels
    if max_label in reject_labels and max_score > min_prob:
        return False
    return max_label in accept_labels and max_score > min_prob

def embed_image(img_path):
    """
    Returns a CLIP embedding vector for the image.
    """
    img = Image.open(img_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
    return features.cpu().numpy()[0]
