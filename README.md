# chunking-transformer

## HierarchicalTransformer (dynamic chunking) appendix code

Split into three files so the architecture can be read independently of the training setup.

model.py: chunk_weights, hard_chunk_bounds, gumbel_hard, extract_chunks), BoundaryPredictor, BatchedAttentionLayer, PositionalEncoding, and HierarchicalTransformer.
data.py: vocab loading, Tokenizer, and ChunkedDataset / get_split / collate for dataset loading
train.py: boundary loss (BoundaryReinforceLoss), optimizer/scheduler setup, training loop with checkpointing and validation, and a sample_from_model for qualitative generation checks. Run directly with python train.py.
### Requirements

torch, pynvml (optional GPU telemetry), unidecode.

### Expected inputs
vocab.txt: one token per line; continuation pieces prefixed ##
data_tokens.txt: whitespace-separated integer token ids
### Outputs
chunk192.pth: checkpoint (model/optimizer/scheduler state), saved every SAVE_STEPS iterations and at the end of each epoch.
loss.txt, valid.txt: plain-text training/validation loss logs.
### Notes for readers

HierarchicalTransformer.forward in model.py is the core of the method; its docstring maps variable names (K, T, D, L, beta, C) to the quantities described in the paper.
