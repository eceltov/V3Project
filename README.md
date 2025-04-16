# V3 Grid Scripts

A collection of scripts intended for the analysis of grid search in image search engines.
The scripts are currently WIP.

## Installation

It is recommended to install the packages and run the scripts from a virtual environment.

```bash
python3 -m venv venv
```

You can activate the environment by running:

```bash
source venv/bin/activate
```

To install dependencies, run:

```bash
python3 -m pip install -r requirements.txt
```

The scripts require a folder with MVK frames and annotations that are not part of this repo.
Please contact the maintainer for the data.
The paths to these folders need to be configured in `config.json` under the `datasetPath` and `annotationsDir` keys.

## Scripts

The following scripts either create embeddings for the analysis, or produce a dataset from these embeddings.

- `datasetCreator.py`: Creates CSV datasets inside the `results` folder (will be created if absent).
Contains a function for each dataset type (annotation, static analysis, dynamic analysis, theoretical analysis).

- `computeTheoreticalEmbeddings.py`: Creates derived embedding files from annotations.
Takes a really long time for a single file (more than an hour).
Due to this, it is intended to be run on multiple nodes.
Takes two arguments; the serial number of the node (0-based) and the total number of nodes running the script.
Each node may be assigned a different number of embeddings to compute, but the script guarantees that two nodes will not work on the same file, as long as they have a different serial number and the same total number of nodes in the launch parameters.
This holds even when the nodes start running at different times.
It is recommended to use a prime numbers for the total number of nodes and change this number to a different prime with every batch to balance the work without much effort (a node is meant more as a job submission rather than a physical machine).
Calling the `debug_print_task_counts()` function will print out how many tasks will be assigned to each node.

- `computeStaticEmbeddings.py`: Creates static segmentation embedding files.
The function call inside the file can be used with different parameters.
Use either "2024" or "2025" for the model year and "whole" or "centerpiece_overlap" for the segmentation kind.

- `computeDynamicEmbeddings.py`: Currently WIP.

The scripts inside the `annotators` folder are the annotation tools used for the study.

The `lib` folder contains various function collections used by the previously mentioned scripts:

- `processingTool.py`: Contains the main infrastructure functions for config loading, annotation retrieval and embedding file I/O.
- The various `analysisTools`: Contain the implementations for how the embeddings are created and datasets derived.
- `boundaries.py`: Contains definitions for various grid segments used by the static analysis.
- `rectangles.py`: Contains several utility functions for rectangle manipulation.

## Usage

The scripts placed directly in the main repo folder can be run as-is.
Make sure you run them from that folder so that the relative paths in `config.json` hold.
