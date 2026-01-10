import torch
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import asyncio
from src.data.dataset import TinyStoriesDataModule
from src.models.transformer import BaselineTransformer
from src.models.contrastive_transformer import ContrastiveTransformer
from src.models.compressor import Compressor
from src.models.linear_model import LinearModel
from src.evaluation.metrics import calculate_metrics
from src.evaluation.llm_eval import LlmEvaluator
from src.evaluation.generation_utils import generate_text
from configs.config import config

# Dynamically import the correct model class based on the model_type

MODEL_MAPPING = {
    'transformer': BaselineTransformer,
    'contrastive': ContrastiveTransformer,
    'compressor': Compressor,
    'linear': LinearModel,
}

def generate_completion(model, tokenizer, input_ids, max_new_tokens=50, temperature=0.7):
    """Generates a text completion from a prompt."""
    output_ids = generate_text(model, tokenizer, input_ids, max_new_tokens, temperature)
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Extract only the newly generated part
    completion = full_text[len(prompt_text):]
    return completion, prompt_text


async def main(args):
    # 1. Load Model and Data
    
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model_class = MODEL_MAPPING.get(args.model_type)
    if not model_class:
        raise ValueError(f"Invalid model_type '{args.model_type}' specified.")
    
    model = model_class.load_from_checkpoint(args.checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.freeze()

    print("Setting up data module and tokenizer...")
    data_module = TinyStoriesDataModule(
        batch_size=1, 
        max_length=config.data.max_length,
        num_workers=0  # Use 0 for evaluation to avoid multiprocessing issues
    )
    data_module.setup('test')
    tokenizer = data_module.tokenizer
    val_dataloader = data_module.val_dataloader()

    # 2. Generate Completions
    
    print(f"Generating {args.num_samples} completions...")
    results = []
    for i, batch in enumerate(tqdm(val_dataloader, total=args.num_samples)):
        if i >= args.num_samples:
            break
        
        input_ids = batch['input_ids']
        
        # Create a prompt from the first half of the story
        prompt_end_idx = torch.sum(batch['attention_mask']) // 2
        prompt_ids = input_ids[:, :prompt_end_idx]
        
        reference_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        completion, prompt_text = generate_completion(
            model, tokenizer, prompt_ids, max_new_tokens=args.max_new_tokens
        )
        
        results.append({
            "prompt": prompt_text,
            "completion": completion,
            "reference": reference_text
        })
    
    # 3. Run Quantitative Metrics
    
    print("Calculating quantitative metrics (BERTScore, ROUGE, etc.)...")
    predictions = [r['completion'] for r in results]
    references = [r['reference'] for r in results]
    
    quantitative_scores = calculate_metrics(predictions, references)
    print("\n--- Quantitative Metrics ---")
    print(json.dumps(quantitative_scores, indent=2))

    # 4. Run LLM-based Evaluation
    
    if args.run_llm_eval:
        print(f"\nRunning LLM-based evaluation with '{args.llm_provider}'...")
        evaluator = LlmEvaluator(provider=args.llm_provider)
        llm_scores = []
        
        tasks = []
        for res in results:
            story_for_eval = f"{res['prompt']} *** {res['completion']}"
            tasks.append(evaluator.evaluate_story(story_for_eval))
        
        llm_results = await asyncio.gather(*tasks)
        
        for i, scores in enumerate(llm_results):
            if scores:
                results[i]['llm_scores'] = scores
        
        # Aggregate LLM scores
        
        df = pd.DataFrame([r['llm_scores'] for r in results if 'llm_scores' in r])
        avg_llm_scores = df.mean().to_dict()
        
        print("\n--- Average LLM Scores ---")
        print(json.dumps(avg_llm_scores, indent=2))
    else:
        avg_llm_scores = {}
    
    # 5. Save all results to a file
    final_output = {
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "quantitative_metrics": quantitative_scores,
        "avg_llm_scores": avg_llm_scores,
        "individual_samples": results
    }

    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{args.model_type}_{os.path.basename(args.checkpoint_path).replace('.ckpt', '')}.json"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nEvaluation complete. Full results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TinyStories model.")
    parser.add_argument("--model_type", type=str, required=True, choices=MODEL_MAPPING.keys(), help="Type of the model to evaluate.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.ckpt file).")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples from the validation set to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate for each completion.")
    parser.add_argument("--run_llm_eval", action='store_true', help="Flag to run the (potentially costly) LLM-based evaluation.")
    parser.add_argument("--llm_provider", type=str, default='openai', choices=['openai', 'google'], help="Which LLM API to use for evaluation.")
    
    args = parser.parse_args()
    asyncio.run(main(args))