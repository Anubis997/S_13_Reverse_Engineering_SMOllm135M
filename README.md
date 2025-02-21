# S_13_Reverse_Engineering_SMOllm135M


In this project, we are trying to reverse engineer SmoLlm2-135M model and train the synthesized model for 5000 steps with generated text and checkpoints for every 500 steps. After 5000 steps, the model has to be run for another 50 steps.

SmoLlm-135M yaml file is avaialble here "https://huggingface.co/HuggingFaceTB/SmolLM2-nanotron-ckpt/blob/main/135M/final/config.yaml"

Here's a high level view of the model:

Parameters: ~135M Attention Heads: 9 (with 3 key-value heads) Activation Function: SiLU (Swish) Vocab Size: 49,152 Sequence Length: 2,048 and Grouped Query Attention.

The original model is trained on Cosmopedia-v2. But, the dataset is too huge with 28 billion tokens. While, it is easier to use online training and train the model, it would take six hours with sequence_length=2048. My Colab's A100 GPU is supporting only 750 tokens at max. With 750 tokens, it would take 2.75 times more time making it sixteen hours and this project we are just trying to understand the model behaivour for 5000 steps. 5000 steps with 2048 sequence length can capture roughly the same context as 13650 steps with 750 sequence length, but that's quite a bit of stretch of simplification we are considering, since, the shorter sequence length means the model is seeing more frequent context truncation. However, if the next batch naturally follows from the previous, the model still retains continuity in learning. But, for the sake of brevity, let's continue with 750 sequence length.

Here's the model that I replicated:

Parameter:134.6M Attention Heads: 9 (with 3 key-value heads) Tokenizer:cosmo2-tokenizer Activation Function SWILU Vocab Size: 49,152 Sequence Length: 750

