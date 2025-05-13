import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pyrallis
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from lavis.models import load_model_and_preprocess
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

    print("Loading BLIP model...")
    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                              is_eval=True, device=device)
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

        with torch.no_grad():
            # extract prompt embeddings
            prompt_features = get_embedding_for_prompt(model, processor, prompt, templates=imagenet_templates)

            # extract blip captions and embeddings
            blip_input_images = [vis_processors["eval"](image).unsqueeze(0).to(device) for image in images]
            blip_captions = [blip_model.generate({"image": image})[0] for image in blip_input_images]
            
            # 使用CLIP processor处理文本
            text_inputs = processor(text=blip_captions, return_tensors="pt", padding=True).to(device)
            caption_embeddings = model.get_text_features(**text_inputs)
            caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)

            text_similarities = [(caption_embedding.float() @ prompt_features.T).item()
                                 for caption_embedding in caption_embeddings]

            results_per_prompt[prompt] = {
                'text_similarities': text_similarities,
                'captions': blip_captions,
                'image_names': image_names,
            }

    # aggregate results
    total_average, total_std = aggregate_text_similarities(results_per_prompt)
    aggregated_results = {
        'average_similarity': total_average,
        'std_similarity': total_std,
    }

    with open(config.metrics_save_path / "blip_raw_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / "blip_aggregated_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_text_similarities(result_dict):
    all_averages = [result_dict[prompt]['text_similarities'] for prompt in result_dict]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std


if __name__ == '__main__':
    run()
