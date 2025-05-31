# Localization Research Project

This repository contains the documentation, results, and code used in the scope of the research project "The influence of localization on video search engine performance".
The goal of this project was to implement a set of tools used to analyze the performance of sub-image search for video search engines utilizing text-to-image similarity models.
Several tools were devised, but generally all tools are computationally intensive and were run for several months on a server dedicated to GPU computation.
The detailed specification of this project can be found [here](./specification.pdf).

# Introduction

CLIP-based text-to-image models form the backbone of modern video search engines.
They allow users to prompt vast video databases with textual queries, yielding accurate results.
However, it is not always possible to formulate effective text queries when searching in video domains requiring expert knowledge.
Competitions like the Video Browser Showdown (VBS) demonstrate this challenge by requiring competitors to search in datasets of medical or marine footage, fueling the development of search techniques that do not rely on text alone.

This study was conducted due to the promising results shown in its pre-study, which focused on sub-image search using a static grid.
Instead of writing prompts for the whole image, grid-based sub-image search allows to specify what part of the image should be searched, removing visual context that is not needed for the query.
Because of the pre-study, an existing video search system called PraK was extended with grid-search functionality, which was subsequently used in the 2025 round of VBS.
The pre-study was summarized in the paper detailing the system [1].

This study conducts a more rigorous investigation into how various sub-image search methodologies perform.
To be able to compare them, a study was held which tasked attendees of various backgrounds to annotate random frames from the Marine Video Kit (MVK) dataset [2].
The collected data served to simulate users using a real video search engine with a simple performance metric that can be measured for all sub-image search methodologies analyzed in this study.
These were, in the order presented in this study, static grid-based localization, textual localization, localization based on object detectors, and a theoretically optimal static localization method that served as the upper performance ceiling.

Two CLIP text-to-image were evaluated in this study, namely the *laion/CLIP-ViT-H-14-laion2B-s32B-b79K* [3] used by the PraK system in the 2024 round of VBS, and *ViT-SO400M-14-SigLIP-384* [4] used in VBS 2025.

# Used Terms

- **Similarity**: A scalar value obtained by computing cosine-similarity on two normalized feature vectors (embeddings).
Higher similarity means that two vectors tend to be more proportional, and thus more likely to be embeddings of similar concepts, whereas lower similarity implies that the vectors are unrelated.
Although different similarity measurements exist, this study uses only cosine-similarity due to its effectiveness and widespread use in the field of video search.
- **Similarity search**: A technique that ranks a set of embeddings based on their similarity with the query embedding.
In video search, this usually takes the form of ranking the individual video frame embeddings based on their similarity to the embedding of a textual user query.
The result of similarity search is a list of frame indices ranked from the most to least similar to the query.
- **Sub-image search**: A specialization of similarity search that utilizes parts of an image instead of the whole image.
- **Grid**, **grid segments**, **grid-search**: A grid is a static partitioning of dataset frames.
The partitions are referred to as grid segments, and grid-search is a realization of sub-image search where frames cropped to a selected grid segment are compared with the query.
Traditional whole-image search can be seen as grid-search using a single grid segment of the size of the whole image for each frame.
This study describes two kinds of grid search; one utilizing a static grid for all frames, and another one that uses a different grid for each frame.
- **Textual localization**: Textual localization is the technique of explicitly adding spatial information into the textual user query.
An example user query without textual localization could be: *A yellow fish*; and with textual localization: *A yellow fish in the upper left corner of the image*.
- **Annotations**: Data collected during the user study.
Each annotation is a composed by an identifier of the annotated frame, a short and long textual description of the annotated object, and the location and dimensions of a bounding box drawn around the object.
In case the annotation was collected with the annotator allowing the user to skip frames, it is referred to as a **skippable** annotation, otherwise it is known as an **non-skippable** annotation.
- **Perturbation**: In this study, perturbation refers to spatial errors in drawn annotation bounding boxes.
While the users annotating frames for this study could draw perfect bounding boxes around objects of interest (OoI), a user operating a video search engine that would allow to specify in what region of the frames the object should be could not draw with such precision.
Such imperfect bounding boxes are considered perturbed.

# User Study

The aim of the user study was to collect MVK annotations with information regarding the locality of annotated objects.
All users were presented with two distinct annotation programs (annotators), the MVK dataset, and textual instructions, which can be found [here](./annotators/INSTRUCTIONS.txt).


Both annotators present the same user interface; their only difference is whether users are allowed to skip frames without annotating any object on them.
The following image is a screenshot of one of the annotators.
The centerpoint of the user interface is a random frame selected from MVK.
By dragging their mouse on the frame, users can draw yellow rectangles to specify what object is being annotated.
This is followed by the annotator presenting the cropped object on the right of the screen, allowing users to double check their annotation.
On the top are two text boxes that the users had to fill out with the short and long descriptions of the object.
Once both the text boxes were filled and a rectangle drawn, the annotation could be submitted.
The user could then either go to the next frame, or annotate another object on the same frame.

![OoI lengths](./figures/annotator.png)

The user study was conducted in two iterations.
Initially, only the annotator that allowed skipping frames was presented to the users, but it was later reasoned that allowing users to skip frames would lead to biases that could not be easily mitigated, such as a lack of annotations of hard to describe objects (i.e., frames containing only rocks).
Therefore, the second annotator was made and distributed to the users.

In total, 12 users partook in the study, half of which had prior experience with localization research, and the other half had no knowledge of the field.
741 unique annotations were collected, 486 of which were obtained with the annotator that allowed to skip frames (skippable annotations), and the remaining 255 were obtained with the annotator that forced the users to annotate at least one object per frame (non-skippable annotations).

The graph below depicts how many annotations were provided per user.

![OoI lengths](./figures/annotation_counts.png)

Although the instructions asked the users to provide exactly 20 non-skippable annotations, a few users provided more.
It was decided that having extra data is more valuable than to enforce consistency, and thus all non-skippable annotations were kept.

Users were asked to provide two textual descriptions for each annotations which different in their lengths.
The annotators showed text hints in the respective description text boxes that indicated how long each description should be, with the shorter one being *short description* (17 characters long), and the longer one *a longer and more detailed description of the selected object* (61 characters, 10 words).
However, because of the following two reasons, the actual length was not enforced.
- Although the MVK dataset contains videos from a single domain, there is a rich variety of objects that users could annotate.
Some objects, usually animals, are much easier to describe than monotonous objects, like rocks.
Forcing a user to write a long description about such monotonous objects could result in descriptions composed primarily of filler words that have close to no weight in search algorithms.
- During the pre-study, where a rougher but still similar annotator was used, it was quickly noted that users tend to become frustrated when asked to provide longer descriptions, especially for monotonous objects.
It would be reasonable to assume that such disgruntled users are less likely to provide more annotations than needed, thus user experience was prioritized.

The following graphs display the lengths of user descriptions in characters and words, respectively.
Although the descriptions are shorter than the text hints on average, this was not considered an issue.
The main reason for requiring two descriptions was to analyze whether longer descriptions improve performance, and not to study their exact relationship.

![OoI lengths](./figures/lengths.png)

Finally, the graph below shows the distribution of annotation bounding boxes.
Although the density slightly differs for the skippable and non-skippable annotations, it is not considered significant.

![Bounding Box Density](./figures/density.png)

# Baseline Analysis

This section will analyze non-localized similarity search, which will serve as a baseline for comparison with the other search methodologies.

## Evaluation

First, it is essential to understand the evaluation process, which is depicted on the figure below.
1. The evaluation algorithm ingests the MVK dataset and a single annotation (in the figure, a user annotated the shark on the protruding frame).
2. Similarity search is conducted with the annotation description, yielding a list of frame indices sorted by similarity.
3. The rank (position) of the annotated frame in the list is identified and used as the final score for this annotation (note that each annotation yields two scores, one for the short description and one for the long one).

This evaluation approach simulates the performance of real video search engines.
In such search engines, the user provides a textual query and is presented with a sorted list of frames most similar to it.
The performance of the search is measured by how long the user has to scroll to find the target frame, analogous to finding the rank of the target (annotation) frame in the sorted list.

<img src="./figures/sim_search.png" alt="Similarity Search" width="1000"/>

## Results

The results will be presented as cumulative graphs with the rank threshold as the x-axis and the percentage of annotations that met this threshold as the y-axis.
If *n* is the number of frames extracted from the MVK dataset, then the x-axis ranges from 0 to *n*.
All graphs will clarify what source data was used for their construction, namely:
- Whether the graph was made with skippable or non-skippable annotations (no graph contains both kinds due to different biases).
- What CLIP text-to-image model was used (either the model used by PraK in year 2025 or 2024 for the VBS competition).
- Whether short of long annotation descriptions were used as queries.
- Whether the annotation bounding boxes were artificially perturbed (this technique will be detailed later).

The graphs below compare the different models and description lengths; the second graph has its x-axis limited to the rank of 100.
Although both graphs contain the same data, the first graph shows that, most of the time, the 2025 model dominates the 2024 model, whereas the cropped graph shows that long descriptions dominate short ones at the beginning of the cumulative function.

![Model and Length Comparison](./figures/model_comparison.png)

A hypothesis was made that this is due to how good queries are.
If the target frame (object) can be uniquely described with a textual query, it will rank at the top even when using weaker models.
Otherwise, the frame will rank somewhere among all other frames for which the query holds, in which case the stronger models prune false positives more effectively.
This claim is supported by the following graphs made from non-skippable annotations.
Notice the difference in the uncropped graph; the long queries using the 2024 model dominate the short queries of the 2025 model much longer than in the case of skippable frames.
Because non-skippable annotations contain harder to describe objects, short descriptions like *two rocks* and *coral* are much more common, and the stronger model struggles to filter out false positives, whereas long queries contain more discernible information that leads to better ranks.

![Model and Length Comparison](./figures/model_comparison_non_skippable.png)

Because this study aims to analyze similarity search methodologies for the purpose of finding good candidates for actual implementation in a video search engine, only a short prefix of the returned lists sorted by similarity will be considered.
The top ranking search results are the most relevant; a video search engine user would not scroll through all 84 thousand returned frames.
Due to this, all following graphs will be bound to the maximum rank of 100.
Subsequently, the y-axis will be bound to 40 % so that all presented graphs are easily comparable.

# Static Grid and Textual Analysis

This section will detail how the static grid analysis was conducted, discuss its results, and compare them to the baseline.
Additionally, textual localization will be analyzed, as is in many ways similar to a static grid.

## Evaluation

The evaluation of the static grid is mostly identical to the baseline evaluation, as shown on the diagram below.
Alongside the dataset and annotation, a specific grid is provided as well.
Before the similarity search is conducted, the intersection-over-union (IoU) is computed for each grid segment with the annotation rectangle, and the segment with the highest IoU is selected; the MVK dataset is then cropped to the dimensions of the selected segment.
Similarity search then uses the cropped MVK dataset to find the rank of the annotated frame.

![Grid Search](./figures/grid_search.png)

Note that this evaluation approach is comparable to the one used on the baseline.
No frames were omitted during similarity search, they were only cropped to a specific region.
This approach will also be applied in the dynamic and theoretical analysis; only a different cropping function will be used.

The evalation process for textual localization is based off the one used for the static grid.
An input grid is provided and IoU is computed; however, instead of cropping the MVK frames, the selected grid segment will be represented with a textual suffix that will be appended to the query.

## Grids

Two types of grids were evaluated; the 5-grid and 9-grid, as seen on the diagram below.

![Grids](./figures/grids.png)

Additional grids with overlapping segments were defined for both grids to test whether they performed better than the base grids.
An example 4-grid with its overlapping version is depicted below.
The 4-grid was chosen for simplicity of presentation; it is not evaluated in this study.

The overlap is measured in percentages relative to the frame sides.
An overlap of 10 % means that each grid segment had its sides prolonged by 10 % of the sides of the whole frame, not the sides of the segment.

![Overlap](./figures/overlap.png)

The following overlaps were measured for the 5- and 9-grids.

- 5-grid: 0 %, 10 %, 20 %, 30 %, and 40 %
- 9-grid: 0 %, 5 %, 10 %, 15 %, 20 %, 25 %, and 40 %.

## Results

The graphs below compare the best performing 5- and 9-grid with the baseline and textual localization.
The 5-grid was chosen for the textual localization, as it still produces reasonably simple suffixes, such as *in the upper left part of the image* or *in the center part of the image"*.

![Static Comparison](./figures/grid_comparison.png)

Both graphs show similar trends, with the second showing significantly worse performance because non-skippable annotations were used.
Also note that there are almost twice as much skippable annotations, resulting in less variance in the first graph.

Using the 9-grid over the baseline resulted in about 66 % more annotations ranking in the first 100 results, which is a surprisingly high performance increase.
Another surprising result is that the textual localization performed worse than the baseline, implying that modern text-to-image models are not strong enough to handle localization on their own.

## Perturbation Results

The graphs above used the perfect, user-drawn annotation bounding boxes to select a grid segment.
However, in a real system, where the user has to draw a bounding box without the image he is trying to find, errors are introduced.
This study attempts to measure the performance decrease with artificial perturbations applied to annotation bounding boxes.

The perturbations were simulated with gaussian noise applied in the form of shifts (moving the bounding box) and deformations (changing the width and height).
Both errors are defined with a single parameter, the *perturbation factor*.
An example perturbation of 0.1 would be translated into the following errors:
- **shift**: The gaussian distribution with a mean of 0 and a standard deviation of 10 (100 times the perturbation factor) will be sampled twice and applied to the vertical and horizontal position of the bounding box.
- **deformation**: The gaussian distribution with a mean of 1 and a standard deviation of 0.1 (the perturbation factor) will be sampled twice and multiplied with the width and height of the bounding box.

For each pair of annotation and perturbation factor, five different bounding boxes were produced to reduce the variation of the results.
The following heatmaps depict the effects of perturbation for the different grid overlap factors and display how many annotations ranked at the top 100 as the recall percentage (effectively the values shown on cumulative graphs at rank 100).
Note that the color scale differs for skippable and non-skippable annotations.

![Grid Perturbations](./figures/heatmaps.png)

The general trend of the graphs is that perturbation decreases the relative performance of grids with smaller overlaps more.
This could be interpreted as grids with bigger overlaps being more rigid; more bounding boxes tend to fall into the correct segments.

The takeaway from these graphs is that systems implementing grid search should consider using higher overlaps in case their users tend to draw bounding boxes with significant perturbation.
An interesting approach would be to dynamically measure the perturbation for each user and employing a grid that best matches their needs on an individual basis.

# Dynamic Analysis

Instead of using a static grid to partition the frames, an object detector could locate objects of interest (OoI) in the dataset and define a custom grid for each frame.
This approach has the significant advantage that each frame can have a different number of segments based on how many OoIs they contain.
Additionally, segment sizes are derived from the OoIs, potentially removing much more visual clutter than static grids.

However, this comes with the significant downside of relying on the object detector to detect relevant object.
Its parameters will have to be fine-tuned to reduce the number of false-positives and false-negatives; and there is always the possibility that the user will search for an object the detector is not trained on.

This study will use the *Grounding DINO* [5] object detector, mainly for its zero-shot detection functionality that does not rely on a predefined list of classes that can be detected.


## Evaluation

The evaluation process uses the same principles as the static grid evaluation.
However, instead of selecting a single segment for the whole dataset based on the IoU with the annotation bounding box, the segments are selected on a per-frame basis.

Because there is no guarantee that there will be a segment with a positive IoU (the object detector could fail to detect anything on a frame), a fallback will be introduced.
Each frame grid will be appended with a segment spanning the whole frame, resulting in every bounding box having positive IoU with at least the fallback segment.
Additionally, because of the size of the fallback segment, its IoU with the bounding box will tend to be small, resulting in other segments with significant intersection to be preferred.

## Results

The following graphs compare dynamic approach with the best performing static grids and the baseline.
Although the same dynamic partitioning was used for both graphs, the first graph shows that the dynamic approach performs about the same as the 5-grid, while the second graph, which uses non-skippable annotations, shows that it starts falling behind.
This is most likely due to the fact that non-skippable annotations contain objects with much lower objectness, such as rocks and corals, which the object detector filtered out.

![Dynamic](./figures/dynamic.png)

Additionally, it was tracked how many times each similarity search had to use the fallback segment for a given frame, meaning there was no detection made by the object detector at the position of the annotation bounding box.
For skippable frames, the fallback segment was used 27.1 % of the time, while for non-skippable ones 25.4 % of the time.
Note that this statistic only depends on the dynamic partitioning and the distribution of the annotation bounding boxes, which is similar for both the skippable and non-skippable ones.

# Theoretical Analysis

Finally, the theoretical analysis aims to find the upper performance bound when using grid-based localization.
Instead of using a static grid for all frames or a different grid for each frame, it utilizes a different grid for each similarity search.
When the user provides the bounding box for the query, the whole dataset is cropped to that box, ensuring an optimal IoU of 100 % for all frames.

The reason that this approach is considered only theoretical is because it is extremely computationally intensive.
In the static grid and dynamic approaches, the grid segment embeddings could be precomputed.
Due to the user-provided bounding box being arbitrary, it is impossible to precompute the embeddings.
Even a performant *NVIDIA H100* takes around 20 minutes to compute the required embeddings for the MVK dataset, making the system unusable for real-time application.

## Evaluation

The evaluation process is effectively the baseline evaluation with a different input dataset.
This dataset is produced by cropping the MVK dataset to the annotation bounding box, as illustrated on the diagram below.

The leftmost column represents the frame the user annotated.
Note that out of the four shark images, only the annotated one remains whole, while two sharks get cropped out entirely.
Whereas the dynamic approach would produce a segment where the shark in the third column would remain whole, this approach crops a significant portion of it away, lowering its final ranking.

![Theoretical Evaluation](./figures/theoretical.png)

## Results

Over the course of several months, the results for the theoretical analysis were computed.
The following graphs compare all methods analyzed in this study.

![Final Analysis](./figures/theoretical_graph.png)

The theoretical approach ranks significantly higher than the other; almost two times better than the baseline for skippable annotations, and over two and a half times better for non-skippable annotations.
Notably, the 9-grid is closer to the theoretical approach than the baseline for skippable frames, although it should be noted that no perturbations were considered in these graphs.

Finally, the last experiment was to test whether enlarging the annotation bounding box improved performance, i.e., whether adding some neighboring visual context helps.
The following graphs shows that it actually significantly harms the performance.

![Box Sizes](./figures/box_size.png)

In this graph, the box enlargement refers to by how many pixels was the annotation bounding box stretched in all four directions.
For reference, all MVK frames have the resolution of 682x384.

# Project

This project is structured as a loose collection of scripts due to the heterogenous nature of the computations.
The general workflow is divided into a long precomputation phase, where the necessary embeddings and detection rectangles are inferred, and a shorter evaluation phase, which yields CSV datasets used for analysis. 

## Installation

It is recommended to install the packages and run the scripts from a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

The scripts require a folder with MVK frames and annotations that are not part of this repository.
Please contact the maintainer for the data.
The paths to these folders need to be configured in `config.json` under the `datasetPath` and `annotationsDir` keys.

## Configuration

To configure the scripts, you can edit values in the `config.json` file.

The most impactful configuration is presented under the *embedConfigs* keys, which dictates what embeddings will be computed and evaluated.
This key can be found in the *derivedDatasetEmbeddings* (theoretical analysis), *staticEmbeddings* (static grid analysis) and *dynamicEmbeddings* (dynamic analysis) sections of the configuration.
In is structured as a list of objects with the following keys:
- *skippable*: What annotations should be used (skippable, non-skippable).
- *model_year*: What text-to-image model should be used.
- *box_enlargements*: For the theoretical analysis, how much should the annotation bounding box be enlarged.
- *kind*: For the static grid analysis, what grids should be used.
- *perturbation_factor*: For the static grid analysis, what perturbation should be applied to the annotation bounding boxes.

The rest of the configuration is mostly paths that do not need to be changed and metadata like frame dimensions.

## Scripts

The following scripts either create embeddings for the analysis, or produce a dataset from these embeddings.

- `datasetCreator.py`: Creates CSV datasets inside the `results` folder (will be created if absent).
Contains a function for each dataset type (annotation, static analysis, dynamic analysis, theoretical analysis).

- `computeTheoreticalEmbeddings.py`: Creates derived embedding files from annotations.
Very computationally intensive, a single annotation file can take several hours to compute.
Due to this, it is intended to be run on multiple nodes.
Takes two arguments; the serial number of the node (0-based) and the total number of nodes running the script.
Each node will have a subset of annotations assigned to it, and the script guarantees that two nodes will not work on the same annotations as long as they have a different serial number and the same total number of nodes in the launch parameters.
This holds even when the nodes start running at different times.
It is recommended to use prime numbers for the total number of nodes and change this number to a different prime with every batch to balance the work (a node is meant more as a job submission rather than a physical machine).

- `computeStaticEmbeddings.py`: Creates static grid embedding files.
The function call inside the file can be used with different parameters.
Use either "2024" or "2025" for the model year and "whole" or "centerpiece_overlap" for the segmentation kind followed with the desired overlap percentage.

- `computeDetectionRects.py`: Uses the Grounding DINO object detector to find objects in frames.
Yields a file with bounding boxes.

- `computeDynamicEmbeddings.py`: Ingests the bounding box file computed in `computeDetectionRects.py` and calculates embeddings for all grid segments.

The scripts inside the `annotators` folder are the annotation tools used for the study.

The `lib` folder contains various function collections used by the previously mentioned scripts:

- `processingTool.py`: Contains the main infrastructure functions for config loading, annotation retrieval and embedding file I/O.
- The various `analysisTools`: Contain the implementations for how the embeddings are created and datasets derived.
- `boundaries.py`: Contains definitions for various grid segments used by the static grid and textual analysis.
- `rectangles.py`: Contains several utility functions for rectangle manipulation, such as IoU computation and rectangle perturbation.

## Usage

The scripts placed directly in the main repository folder can be run as-is.
Make sure you run them from that folder so that the relative paths in `config.json` hold.

Note that the `computeDetectionRects.py` script requires the Grounding DINO package, that is installed directly from the Grounding DINO repository found [here](https://github.com/IDEA-Research/GroundingDINO).

# References
[1] Stroh, Michael; Kloda, Vojtěch; Verner, Benjamin; et al. PraK Tool V3: Enhancing Video Item Search Using Localized Text and Texture Queries. In: Ide, Ichiro; Kompatsiaris, Ioannis; Xu, Changsheng; Yanai, Keiji; Chu, Wei-Ta; Nitta, Naoko; Riegler, Michael; Yamasaki, Toshihiko (eds.). MultiMedia Modeling. Singapore: Springer Nature Singapore, 2025, pp. 326–333. isbn 978-981-96-2074-6.

[2] Truong, Quang-Trung; Vu, Tuan-Anh; Ha, Tan-Sang; Jakub, Lokoc; Tim, Yue Him Wong; Joneja, Ajay; Yeung, Sai-Kit. Marine Video Kit: A New Marine Video Dataset for Content-based Analysis and Retrieval. 2022. Available from arXiv: 2209.11518 [cs.CV].

[3] laion/CLIP-ViT-H-14-laion2B-s32B-b79K. Available also from: https://https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

[4] Retrieval Optimized CLIP Models. Available also from: https://github.com/Visual-Computing/MCIP

[5] Liu, Shilong; Zeng, Zhaoyang; Ren, Tianhe; Li, Feng; Zhang, Hao; Yang, Jie; Li, Chunyuan; Yang, Jianwei; Su, Hang; Zhu, Jun, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499. 2023