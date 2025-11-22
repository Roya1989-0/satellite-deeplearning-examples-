# satellite-deeplearning-jahrom-# Tree Counting in Jahrom Orchards (Dummy Example)

This example demonstrates a minimal deep-learning pipeline for **tree counting**
in orchards around Jahrom using *dummy* satellite-like image patches.

The goal is to show how one could:

- Structure a project for tree counting from remote sensing imagery
- Implement a simple CNN-based regression model in PyTorch
- Work with a custom Dataset that returns `(image, tree_count)` pairs
- Track loss and MAE over epochs


## Project Structure

- `train_tree_counter_dummy.py` – CNN model, dummy dataset, training loop
- `requirements.txt` – Python dependencies

## How to Run

```bash
pip install -r requirements.txt
python train_tree_counter_dummy.py
