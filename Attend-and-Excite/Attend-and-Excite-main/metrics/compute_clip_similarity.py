import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pyrallis
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")
    clip_model_path: str = "G:/code/model/clip-vit-base-patch16"

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model = CLIPModel.from_pretrained(config.clip_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(config.clip_model_path)
    model.eval()
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}
    for prompt in tqdm(prompts):
        print(f'Running on: "{prompt}"')

        # get all images for the given prompt
        image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
        images = [Image.open(p) for p in image_paths]
        image_names = [p.name for p in image_paths]
        
        # 使用CLIP processor处理图像
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # split prompt into first and second halves
            if ' and ' in prompt:
                prompt_parts = prompt.split(' and ')
            elif ' with ' in prompt:
                prompt_parts = prompt.split(' with ')
            else:
                print(f"Unable to split prompt: {prompt}. "
                      f"Looking for 'and' or 'with' for splitting! Skipping!")
                continue

            # extract texture features
            full_text_features = get_embedding_for_prompt(model, processor, prompt, templates=imagenet_templates)
            first_half_features = get_embedding_for_prompt(model, processor, prompt_parts[0], templates=imagenet_templates)
            second_half_features = get_embedding_for_prompt(model, processor, prompt_parts[1], templates=imagenet_templates)

            # extract image features
            images_features = model.get_image_features(**image_inputs)
            images_features = images_features / images_features.norm(dim=-1, keepdim=True)

            # compute similarities
            full_text_similarities = [(feat.float() @ full_text_features.T).item() for feat in images_features]
            first_half_similarities = [(feat.float() @ first_half_features.T).item() for feat in images_features]
            second_half_similarities = [(feat.float() @ second_half_features.T).item() for feat in images_features]

            results_per_prompt[prompt] = {
                'full_text': full_text_similarities,
                'first_half': first_half_similarities,
                'second_half': second_half_similarities,
                'image_names': image_names,
            }

    # aggregate results
    aggregated_results = {
        'full_text_aggregation': aggregate_by_full_text(results_per_prompt),
        'min_first_second_aggregation': aggregate_by_min_half(results_per_prompt),
    }

    with open(config.metrics_save_path / "clip_raw_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / "clip_aggregated_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_per_half_res = [[min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)


def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


if __name__ == '__main__':
    run()
