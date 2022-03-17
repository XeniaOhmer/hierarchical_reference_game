# hierarchical_reference_game


Implementation of a hierarchical reference game using [EGG](https://github.com/facebookresearch/EGG/). The code belongs to the manuscript: "Emergence of hierarchical reference systems in multi-agent communication".

For the sake of coherence, the paper describes the relevance vector as indicating relevance by "1" and irrelevance by "0". Without loss of generality the implementation reverses this encoding (0:relevant, 1:irrelevant).


#### Agent training 
Agents can be trained using 'train.py'. The file provides explanations for how to configure agents and training using command line parameters. 

#### Evaluation
Our results can be found in 'results/'. The subfolders contain the metrics for each run. Some metrics were stored during training by saving the callback outputs. In addition, we stored the final interaction for each run. The interaction logs all game-relevant information such as sender input, messages, receiver input, and receiver selections for the training and validation set. Based on these interactions, we evaluated additional metrics after training using the notebook 'evaluate_metrics.ipynb'. We uploaded all metrics but not the interaction files due to their large size.

#### Visualization
Visualizations of training and results, including all results and figures included in the paper, can be found in the notebooks 'analysis.ipynb' and 'analysis_control.ipynb'. The former reports all results for the 6 different data sets (*D(3,4)*, *D(3,8)*, *D(3,16)*, *D(4,4)*, *D(4,8)*, *D(5,4)*) with unbalanced distractor sampling. The latter reports all results for our control experiements with different distractor sampling strategies and different vocabulary sizes, only for data set *D(4,8)*. 

#### Grid search results 
The folder 'grid_search/' contains the results for the grid search, as well as a pdf summarizing the results. 


