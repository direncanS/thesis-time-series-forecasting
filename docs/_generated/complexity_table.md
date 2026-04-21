| Model | n_params | Architectural category | Wall-clock seconds | Hardware note |
|-------|----------|------------------------|---------------------|----------------|
| LR    | 113 064 | linear                 | < 1 s               | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| MLP   | 124 328 | shallow-MLP            |                     | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| LSTM  |  29 608 | recurrent              |                     | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
| TFT   |  18 261 | transformer-family     |                     | single NVIDIA RTX 5080 Laptop GPU (CUDA), batch_size = 64 |
