# IVF-PQ Semantic Search

This repository implements an efficient and scalable **Semantic Search Engine** using **Inverted File (IVF)** and **Product Quantization (PQ)** techniques. The project explores approximate nearest neighbor (ANN) search with FAISS to handle large-scale vector databases.

## Overview

The IVF-PQ approach enables efficient search in large vector datasets by partitioning the search space and applying quantization techniques for memory-efficient storage and retrieval. This project uses **cosine similarity** for vector comparisons and supports databases of up to 20 million records (~5GB).

## Features

- **Inverted File (IVF)**: Efficient partitioning of the vector space to reduce search complexity.
- **Product Quantization (PQ)**: Compact representation of vectors for efficient storage.
- **Cosine Similarity**: Accurate vector comparisons for semantic search.

## Directory Structure

```
IVF-PQ-Semantic-Search/
├── lvfTrain.py            # Main training script
├── evaluation.py          # Evaluation script
├── vec_db.py              # Vector database management
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedZahran02/IVF-PQ-Semantic-Search.git
   cd IVF-PQ-Semantic-Search
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```bash
   python lvfTrain.py
   ```

2. Evaluate the search engine:
   ```bash
   python evaluation.py
   ```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

