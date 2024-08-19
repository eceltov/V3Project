# V3 Grid Scripts

A collection of scripts intended for the analysis of grid search in image search engines.

## Scripts

- `selectiveDescriber.py`: A window app that allows the user to select portions of frames and annotate them with text queries that could be used to search for the frame in the search engine.
Each annotation is saved in the `jobs.json` file.

- `layoutAnalyzer.py`: A script which goes through the annotations in `jobs.json` and analyzes the performance of different grids.
The script finds the grid segment with the biggest overlap to the annotated frame segment and calculates its position relative to all other frames for that segment.
This simulates the user finding an object of interest in the frame, writing a descriptive text query, and selecting a grid segment which contains the most of the object.

- `featureCreator.py`: A script for creating grid features used by the other scripts.

- `jobExecutor.py`: A script which goes through all jobs in `jobs.json`, extracts its bounding rectangles, and for each of these rectangles creates a feature set for the whole frame database where the frames are cropped to the bounding rectangle, effectively hiding all parts of frames that were outside of the rectangle.
This is used for analysis whether text queries perform better in smaller frame segments filled with objects of interest, or whether the extended frame context helps the search engine.
This script takes a long time to execute due to the amount of features it needs to create.
It will cache created features in the `features` folder (create it if not present).

- `reverseSegmentSearch.py`: A script which uses the reversed grid image search method to evaluate the performance of different grids in image queries.

- `jobManager.py`: A library containing functions used by other scripts.

## Usage

First set up the path to the frame dataset in `config.json`.

It should be possible to run all scripts (except `jobManager.py`, because it is a library) as is.

