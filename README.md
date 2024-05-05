# Fine-Tuning Framework for Language Models
### A Fine-Tuning framework for Language Models to enhance the quality of Text Embeddings for Classification and Semantic Textual Similarity
 
Here, we have created a language model fine-tuning framework which is very intuitive and easy to use. We have control over tokenization, pooling, data collation and loss functions to train and test on two different types of tasks, classification and semantic textual similarity.

__________
### For now this framework fine-tunes all the BERT-based models like BERT-base, BERT-large, RoBERTA, SBERT etc. Further update will include fine-tuning LLMs like Llama.

__________

Every step from fine-tuning to creating performance result plots have been compiled in two notebook files:

**fine_tuning_bert_models_SentEval_STS.ipynb**: For fine-tuning and results matrix export.

**results_plots_senteval_sts.ipynb**: For creating performance plots for the generated results matrix.
__________

### Model Selection:

Enter your model names present in HuggingFace Library in the "models" array:
![Models](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/9c48082d-2654-4c3c-8be5-ef2967d7231c)

### Dataset Selection:

#### Classification:
The framework has been trained and tested for classification tasks using Movie Reviews(MR), Customer Reviews(CR), Multi-Perspective Question Answering (MPQA) and Subjective vs Objective(SUBJ) datasets from SentEval. After fine-tuning, the model generates embeddings of the sentences in the train and test sets, and uses a Logistic Regression model to train and test using the embedding vector as feature set and the labels for classification. Any dataset should be a Pandas Dataframe having at least the following two columns:

**sentence**: This column should have the list of sentences for processing and training.

**label**: This column will contain the labels for classification.

The dataset should look something like this:

![Classification_Dataset](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/27555f30-27d8-4ae7-81f0-b62a8b990b8a)

_________
The following method is used for classification task to get the results matrix of models using all the loss function combinations:

**driver_senteval()**

Here, a list of SentEval datasets have been provided for training and evaluation, which are obtained using the **get_senteval_dataset(name)** function and their names have been passed to the **senteval_datasets** array (MR, CR, MPQA and SUBJ). For custom datasets, you need to create a similar importer function following the column specifications mentioned above, pass the names of the datasets in the **senteval_datasets** array and pass the custom importer function in the **dataset preparation** stage of **driver_senteval()** instead of **get_senteval_dataset(name)**.
_________

#### Semantic Textual Similarity:
The framework has been trained and tested for sentence similarity tasks using STS12-16, STS-Benchmark and SICK-R datasets from SemEval. The fine-tuned model will generate embeddings of the test sentence pair, calculate the cosine simlarity between the embeddings of each sentence pair, and finally calculate the Spearman's Rank Correlation Coefficient. Any dataset should be a Pandas Dataframe having at least the following three columns:

**sentence1** and **sentence2**: These columns should have the list of sentence pairs for processing and training.

**score**: This column will contain the similarity scores.

The dataset should look something like this:

![STS_Dataset](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/f4ab8ad2-cac8-4c00-bafd-33f3e9a2d74d)

_________
The following method is used for sentence similarity task to get the results matrix of models using all the loss function combinations:

**driver_sts()**

Here, the STS datasets are obtained using the **get_sts_dataset(name)** function and their names have been passed to the **sts_datasets** array (STS12-16, STS-B and SICK-R). For custom datasets, you need to create a similar importer function following the column specifications mentioned above, pass the names of the datasets in the **sts_datasets** array and pass the custom importer function in the **dataset preparation** stage of **driver_sts()** instead of **get_sts_dataset(name)**.

__________
The **results matrix** will include the corresponding classification accuracy or Spearman's rank correlation coefficient values for all models across all datasets using all possible combinations of loss functions.
__________

### Available Loss Functions:

1. **Cosine Loss**: End-to-end optimization of Cosine Similarity between vectors.
2. **In-Batch Negatives Loss**: The in-batch negative samples in a batch which are identical but not explicitly labelled as positive, are identified and labelled as positive, thus reducing noise.
3. **Angle Loss**: Both the other loss functions are dependant on cosine, and cosine has saturation zones where training becomes extremely slow. To mitigate this, an optimization strategy is used which optimizes the angle difference in complex space.

The weight combinations of each loss function can be modified. For example (1, 0, 0) means that only the Cosine Loss will be activated with weight = 1. A weight of 0 will exclude the loss function:
![Loss](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/677fac1a-4a14-430e-9f8f-f883bb37fd5a)

The temperature hyperparameters for each loss function (**cosine_tau**, **ibn_tau** and **angle_tau**) can be modified in the **loss_kwargs** along with their individual weights:

![Loss_kwargs](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/797476c8-010e-4162-8cbc-9512ae7200fc)

The results matrix can be exported into .npy files once the process is complete, which can be fed to the **results_plots_senteval_sts.ipynb** file for plotting the results. This gives an insight about the overall model performances across all datasets using all possible loss function combinations.
