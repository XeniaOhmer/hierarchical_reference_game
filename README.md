# hierarchical_reference_game


Implementation of a hierarchical reference game using [EGG](https://github.com/facebookresearch/EGG/). The code belongs to the manuscript: "Emergence of hierarchical reference systems in multi-agent communication".

For the sake of coherence, the paper describes the relevance vector as indicating relevance by "1" and irrelevance by "0". Without loss of generality the implementation reverses this encoding (0:relevant, 1:irrelevant).


#### Agent training 

Agents can be trained using 'train.py'. The file provides explanations for how to configure agents and training using command line parameters. 

For example, to train the agents on data set *D(4,8)* (4 attributes, 8 values), with vocab size factor 3 (default), with the same hyperparameters as in the paper, you can execute

    python train.py --dimensions 8 8 8 8 --n_epochs 300 --batch_size 32
    
Similarly, for data set *D(3, 4)*, the dimensions flag would be 

    --dimensions 4 4 4

Per default, this conducts one run. If you would like to change the number of runs, e.g. to 5, you can specify that using 

    --num_of_runs 5
    
If you would like to save the results (interaction file, agent checkpoints, a file storing all hyperparameter values, training and validation accuracies over time, plus test accuracy for generalization to novel objects) you can add the flag

    --save True 
    
If you would like to change the vocab size factor as in the control experiments, e.g. to 1, you can add the flag 

    --vocab_size_factor 1
    
Or if you would like to change the distractor sampling strategy to "balanced" as in the control experiments, use

    --balanced_distractors True
    
    
To retrain the agents and evaluate them on zero-shot generalization to novel abstractions, you can execute

    python train.py --dimensions 8 8 8 8 --zero_shot True --n_epochs 300 --batch_size 32 --num_of_runs 1

  

#### Evaluation

Our results can be found in 'results/'. The subfolders contain the metrics for each run. Some metrics were stored during training by saving the callback outputs. In addition, we stored the final interaction for each run. The interaction logs all game-relevant information such as sender input, messages, receiver input, and receiver selections for the training and validation set. Based on these interactions, we evaluated additional metrics after training using the notebook 'evaluate_metrics.ipynb'. We uploaded all metrics but not the interaction files due to their large size.

#### Visualization

Visualizations of training and results, including all results and figures included in the paper, can be found in the notebooks 'analysis.ipynb' and 'analysis_control.ipynb'. The former reports all results for the 6 different data sets (*D(3,4)*, *D(3,8)*, *D(3,16)*, *D(4,4)*, *D(4,8)*, *D(5,4)*) with unbalanced distractor sampling. The latter reports all results for our control experiements with different distractor sampling strategies and different vocabulary sizes, only for data set *D(4,8)*. 

#### Grid search results 

The folder 'grid_search/' contains the results for the grid search, as well as a pdf summarizing the results. 


