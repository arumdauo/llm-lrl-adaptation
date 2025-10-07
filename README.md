# llm-lrl-adaptation
Adapting LLMs to low-resource languages (Tamil, Yoruba, Nepali) via Grapheme-aware PickyBPE and SaGe tokenization, vocabulary expansion, and LoRA-based continual pre-training.

## Features
- **Custom Tokenizers**: PickyBPE (removes redundant tokens during training), GraphemePickyBPE (grapheme-aware for complex scripts) and SaGe (context-based pruning)
- **Vocabulary Expansion**: Three initialization strategies (random, mean, merge-based)
- **Parameter-Efficient Training**: LoRA-based continual pre-training for LLaMA 2 7B and Gemma 7B

## Project Structure (language folder example – same for all languages)
```bash
<language>/
├── 1_preprocessing/
│   ├── evaluate_tokenizer.ipynb
│   ├── picky_bpe.ipynb
│   ├── sage.ipynb
│   └── split_corpus.ipynb
│
├── 2_instantiation/
│   ├── gemma_mean_init.ipynb
│   ├── gemma_merge_init.ipynb
│   ├── gemma_random_init.ipynb
│   ├── llama2_mean_init.ipynb
│   ├── llama2_merge_init.ipynb
│   └── llama2_random_init.ipynb
│
├── 3_lapt/
│   ├── gemma_init_lapt.ipynb
│   └── llama2_init_lapt.ipynb
│
└── 4_evaluation/
    └── eval_sum.ipynb
```

### 1. Preprocessing

Trains custom tokenizers on target language corpora and evaluates them using fertility and compression metrics.

- `split_corpus.ipynb`: Data preparation and train/eval splits
- `picky_bpe.ipynb`: PickyBPE/GraphemePickyBPE training
- `sage.ipynb`: SaGe tokenizer training
- `evaluate_tokenizer.ipynb`: Intrinsic tokenizer evaluation

### 2. Instantiation

Expands base model vocabularies (~10K new tokens) with different embedding initialization strategies.

- `*_random_init.ipynb`: Random embedding initialization
- `*_mean_init.ipynb`: Mean-based initialization (selected method for LAPT)
- `*_merge_init.ipynb`: BPE merge hierarchy-based initialization

### 3. LAPT

Applies LoRA-based continual pre-training on target language data.

- `*_init_lapt.ipynb`: Language-Adaptive Pre-Training with CLM objective

### 4. Evaluation

Evaluates adapted models on intrinsic (perplexity) and extrinsic (summarization) tasks.

- `eval_sum.ipynb`: Zero-shot and few-shot summarization using XL-Sum dataset

## References
This project implements methods from:
- **PickyBPE**: Chizhov, P., Arnett, C., Korotkova, E., & Yamshchikov, I. P. (2024). "BPE gets picky: Efficient vocabulary refinement during tokenizer training." *arXiv preprint arXiv:2409.04599*. [[paper]](https://arxiv.org/abs/2409.04599)
- **Grapheme Pair Encoding**: Velayuthan, M., & Sarveswaran, K. (2024). "Egalitarian language representation in language models: It all begins with tokenizers." *arXiv preprint arXiv:2409.11501*. [[paper]](https://arxiv.org/abs/2409.11501)
- **SaGe**: Yehezkel, S., & Pinter, Y. (2022). "Incorporating context into subword vocabularies." *arXiv preprint arXiv:2210.07095*. [[paper]](https://arxiv.org/abs/2210.07095)

## Data
The data for this project is available on [Hugging Face Hub](https://huggingface.co/datasets/astasol/llm-lrl-adaptation).





