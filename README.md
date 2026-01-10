# A Comprehensive Analysis of Small Language Models on TinyStories

This project presents a comparative study of different language model architectures and training techniques on the TinyStories dataset. It explores the trade-offs between model complexity, performance, and efficiency, questioning the necessity of large-scale transformers for generating coherent narratives in a constrained domain.

## Core Research Questions

- How do traditional Transformers compare to simpler, non-attention-based models like MLPs?
- Can novel training methods like contrastive learning improve narrative coherence?
- Can architectural modifications reduce the memory footprint of Transformers?
- How do objective metrics (BERTScore, ROUGE) compare to subjective LLM-based evaluations?

## Models Explored

This project implements and compares four distinct approaches:

1. **Baseline Transformer**: A from-scratch, decoder-only Transformer serving as our baseline.
2. **Contrastive Transformer**: Extends the baseline by introducing a contrastive loss term to explicitly penalize incoherent sentence ordering.
3. **Compressor Model**: A novel, memory-efficient architecture that compresses token pairs before feeding them into a central Transformer trunk, reducing the sequence length.
4. **Linear Model**: A simple autoregressive MLP that replaces the self-attention mechanism with masked linear layers, testing the power of autoregression itself.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/siddharthdhara/slm-tinystories.git
   cd slm-tinystories
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Training

The unified training script `src/training/train.py` can be used to train any of the four models. Use the `--model_type` argument to select the model.

#### Train the Baseline Transformer:
```bash
python -m src.training.train --model_type transformer
```

#### Train the Contrastive Transformer:
```bash
python -m src.training.train --model_type contrastive
```

#### Train the Compressor Model:
```bash
python -m src.training.train --model_type compressor
```

#### Train the Linear Model:
```bash
python -m src.training.train --model_type linear
```

### Evaluation

Use the `src/evaluation/evaluate.py` script to run evaluations on a trained model checkpoint.

```bash
python -m src.evaluation.evaluate --checkpoint_path checkpoints/transformer/best_model.ckpt --eval_type all
```

## Repository Structure

```
├── configs/
│   └── config.py                           # Central configuration file for all hyperparameters
├── figures/                                # Training and validation plots
│   ├── chkpt-train-batch.png
│   ├── chkpt-train.png
│   ├── chkpt-validate-batch.png
│   ├── chkpt-validate.png
│   ├── sanity-loss-batch-train.png
│   ├── sanity-loss-batch-validate.png
│   ├── sanity-loss-train.png
│   ├── sanity-loss-validate.png
│   └── sanity-loss.png
├── graphs/                                 # Model comparison and analysis plots
│   ├── compression.png
│   ├── contrastive.png
│   ├── heads.png
│   ├── hidden_dim.png
│   ├── instruct.png
│   ├── layers.png
│   ├── pretrained.png
│   ├── TrainLoss.png
│   └── ValLoss.png
├── interpret_vis/                          # Attention visualization outputs
│   └── attention_head_*.png                # Attention head visualizations for different configurations
├── neuron_analysis/                        # Neuron activation analysis
│   ├── 1-Neuron2-adjs.png
│   └── 4-Neuron1-nouns.png
├── notebooks/
│   └── analysis_and_visualization.ipynb    # Example Jupyter notebook for results analysis and visualization (Not the same one we used)
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py                  # Baseline Transformer implementation
│   │   ├── contrastive_transformer.py      # Contrastive Transformer with additional loss term
│   │   ├── compressor.py                   # Memory-efficient compressor model
│   │   └── linear_model.py                 # Simple autoregressive MLP model
│   ├── data/
│   │   └── dataset.py                      # Data loading and preprocessing utilities
│   ├── training/
│   │   └── train.py                        # Unified training script for all models
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py                     # Main evaluation script
│       ├── evaluate_gpt2_pretrained.py     # GPT-2 pretrained model evaluation
│       ├── evaluate_instruct.py            # Instruction-tuned model evaluation
│       ├── evaluate_selftrained.py         # Self-trained model evaluation
│       ├── interpretability.py             # Model interpretability analysis
│       ├── krct_prompt_eval.py            # KRCT prompt-based evaluation
│       ├── llm_eval.py                    # LLM-based evaluation methods
│       ├── metrics.py                     # Objective metrics (BERTScore, ROUGE)
│       ├── model_compare.py               # Model comparison utilities
│       ├── rouge_scores.py                # ROUGE score calculations
│       ├── run_compressor.py              # Compressor model evaluation runner
│       ├── run_gpt2.py                    # GPT-2 evaluation runner
│       └── run_gpt2_cont.py               # GPT-2 continued evaluation
├── Project Report.docx                     # Comprehensive project report
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```