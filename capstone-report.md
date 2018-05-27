# Machine Learning Engineer Nanodegree
## Capstone Project
Akihiro Nakao
March 21st, 2018

## Proposal Link

https://review.udacity.com/#!/reviews/1242504

## I. Definition
### Domain Background and Problem Statement

At the laboratory called CERN, we are doing research on proton collision to elucidate the origins of the universe.
However, in this proton study, a large amount of data is generated and physicists have to process about 10 petabytes of data per year.

When elementary particles collide with a detector, coordinates in three dimensions can be obtained. Actually this coordinates physicist need to figure out the trajectory of the particle.

In other words, it is necessary to join the trajectories related to certain particles based on the large amount of coordinate data obtained.

However, it is impossible to join the orbits actually by hand, as there is huge data.So we use machine learning to predict this trajectory. Physicists can concentrate on their duties by predicting orbit.


`Predict trajectory by machine learning`
[![https://diveintocode.gyazo.com/fdddb2f1339e30a0c4168b105a075d1f](https://t.gyazo.com/teams/diveintocode/fdddb2f1339e30a0c4168b105a075d1f.png)](https://diveintocode.gyazo.com/fdddb2f1339e30a0c4168b105a075d1f)



`CERN official site`

https://sites.google.com/site/trackmlparticle/

`kaggle`

https://www.kaggle.com/c/trackml-particle-identification

#### Datasets and Inputs

A dataset comprises multiple independent events, where each event contains simulated measurements (essentially 3D points) of particles generated in a collision between proton bunches at the Large Hadron Collider at CERN.

https://www.kaggle.com/c/trackml-particle-identification/data

These data sets contain the following information:

- Event hits
- Event truth
- Event particles
- Event hit cells

The data is divided into the above data, and when all the data are combined, there are 11 characteristic quantities. It also consists of 8850 rows. This data can be handled by a dedicated library trackML.

https://github.com/LAL/trackml-library


#### task
To predict the trajectory of certain elementary particles. Although the data set contains data on the trajectories of many elementary particles, it is unknown which elementary particle the data is on in the orbit of some elementary particle.

#### output
Identify which elementary particle trajectory the input data belongs to (Classification problem)


### Solution Statement

For this time, we can make prediction using DBSCAN which is a type of clustering algorithm. By using the clustering algorithm, we can predict which orbit the data belongs to.

In addition, we can measure and verify this prediction with Custom Metric.

`detail`

![](/Users/akihiro/udacity/machine-learning-capstone/report-images/DBSCAN-1.png)

DBSCAN is a density-based clustering algorithm. As shown in the illustration, it is expected that the density of each particle will be smaller if it is the same particle trajectory. DBSCAN can cluster in multidimensional without specifying how many clustering is done beforehand.

https://arxiv.org/pdf/1012.6009.pdf


### Benchmark Model

I used the knn approach to check if the problem can be solved.The model with the local score of 0.09900 will be used as benchmark model.

`The following code is forked from the kernel`
https://www.kaggle.com/lasershow/knn-approach/code

### Evaluation Metrics

`summary`
The evaluation metric for this competition is a custom metric.
Each hit is weighted and the score is calculated as follows.
track is uniquely matched to particles by double majority rule. Match and sum the remaining tracks and make it a total of events.

`detail`
The evaluation metric for this competition is a custom metric. In one line : it is the intersection between the reconstructed tracks and the ground truth particles, normalized to one for each event, and averaged on the events of the test set.

First, each hit is assigned a weight:

the few first (starting from the center of the detector) and last hits have a larger weight
hits from the more straight tracks (more rare, but more interesting) have a larger weight
random hits or hits from very short tracks have weight zero
the sum of the weights of all the hits of one event is 1 by construction
the hit weights are available in the truth file. They are not revealed for the test dataset

Then, the score is constructed as follows:

tracks are uniquely matched to particles by the double majority rule:
for a given track, the matching particle is the one to which the absolute majority (strictly more that 50%) of the track points belong.
the track should have the absolute majority of the points of the matching particle. If any of these constraints is not met, the score for this track is zero
the score of a surviving track is the sum of the weights of the points of the intersection between the track and the matching particle.
the score of an event is the sum of the score of all its tracks.

`justification`
In this contest, we need to use dedicated metrics in elementary particle physics, not metrics that are normally used to properly evaluate problems in elementary particle physics.

https://www.kaggle.com/c/trackml-particle-identification#evaluation

https://github.com/LAL/trackml-library



## II. Analysis

### Data Exploration

First of all, I grasp the characteristics of the data mainly using the describe method.

`hits`
[![https://diveintocode.gyazo.com/246f36c83c4b2daeaf86595dc9364263](https://t.gyazo.com/teams/diveintocode/246f36c83c4b2daeaf86595dc9364263.png)](https://diveintocode.gyazo.com/246f36c83c4b2daeaf86595dc9364263)

The x, y (in millimeter) coordinates range from about -1050 to 1050, respectively. The z coordinate ranges from about -3000 to 3000

`cells`
[![https://diveintocode.gyazo.com/fc80f5cde88b70f047ff85be0f612cce](https://t.gyazo.com/teams/diveintocode/fc80f5cde88b70f047ff85be0f612cce.png)](https://diveintocode.gyazo.com/fc80f5cde88b70f047ff85be0f612cce)

`particles`
[![https://diveintocode.gyazo.com/27b9bfd3e100b4c32fe179d860ef6359](https://t.gyazo.com/teams/diveintocode/27b9bfd3e100b4c32fe179d860ef6359.png)](https://diveintocode.gyazo.com/27b9bfd3e100b4c32fe179d860ef6359)

`truth`
[![https://diveintocode.gyazo.com/68e92e9fbce953b79fb76e92aecc02dd](https://t.gyazo.com/teams/diveintocode/68e92e9fbce953b79fb76e92aecc02dd.png)](https://diveintocode.gyazo.com/68e92e9fbce953b79fb76e92aecc02dd)


### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

![](/Users/akihiro/udacity/machine-learning-capstone/report-images/trackml-img-1.png)

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
