# Fine-Tuning Framework for Language Models
### A Fine-Tuning framework for Language Models to Improv Quality of Text Embeddings for Classification and Semantic Textual Similarity
 
Here, we have created a language model fine-tuning framework which is very intuitive and easy to use. We have control over tokenization, pooling, data collation and loss functions to train and test on two different types of tasks, classification and semantic textual similarity.

### Model Selection:

Enter your model names present in HuggingFace Library in the "models" array:
![Models](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/9c48082d-2654-4c3c-8be5-ef2967d7231c)

### Dataset Selection:

#### Classification:
The framework has been trained and tested for classification tasks using Movie Reviews(MR), Customer Reviews(CR), Multi-Perspective Question Answering (MPQA) and Subjective vs Objective(SUBJ) datasets from SentEval. After fine-tuning, the model generates embeddings of the sentences in the train and test sets, and uses a Logistic Regression model to train and test using the embedding vector as feature set and the labels for classification. Any dataset should be a Pandas Dataframe having at least the following two columns:

**sentence**: This column should have the list of sentences for processing and training.

**label**: This column will contain the labels for classification.

Run the following method for classification task to get the results matrix of models using all the loss function combinations:

**driver_senteval()**

Here, a list of SentEval datasets have been provided for training and evaluation, which are obtained using the **get_senteval_dataset(name)** function and their names have been passed to the **senteval_datasets** array (MR, CR, MPQA and SUBJ). For custom datasets, you need to create a similar importer function following the column specifications mentioned above, pass the names of the datasets in the **senteval_datasets** array and pass the custom importer function in the **dataset preparation** stage of **driver_senteval()** instead of **get_senteval_dataset(name)**.

#### Semantic Textual Similarity:
The framework has been trained and tested for sentence similarity tasks using STS12-16, STS-Benchmark and SICK-R datasets from SemEval. The fine-tuned model will generate embeddings of the test sentence pair, calculate the cosine simlarity between the embeddings of each sentence pair, and finally calculate the Spearman's Rank Correlation Coefficient. Any dataset should be a Pandas Dataframe having at least the following three columns:

**sentence1** and **sentence2**: These columns should have the list of sentence pairs for processing and training.

**score**: This column will contain the similarity scores.

Run the following method for sentence similarity task to get the results matrix of models using all the loss function combinations:

**driver_sts()**

Here, the STS datasets are obtained using the **get_sts_dataset(name)** function and their names have been passed to the **sts_datasets** array (STS12-16, STS-B and SICK-R). For custom datasets, you need to create a similar importer function following the column specifications mentioned above, pass the names of the datasets in the **sts_datasets** array and pass the custom importer function in the **dataset preparation** stage of **driver_sts()** instead of **get_sts_dataset(name)**.

### Available Loss Functions:

1. **Cosine Loss**: End-to-end optimization of Cosine Similarity between vectors.
2. **In-Batch Negatives Loss**: The in-batch negative samples in a batch which are identical but not explicitly labelled as positive, are identified and labelled as positive, thus reducing noise.
3. **Angle Loss**: Both the other loss functions are dependant on cosine, and cosine has saturation zones where training becomes extremely slow. To mitigate this, an optimization strategy is used which optimizes the angle difference in complex space.

The weight combinations of each loss function can be modified. For example (1, 0, 0) means that only the Cosine Loss will be activated with weight = 1. A weight of 0 will exclude the loss function:
![Loss](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/677fac1a-4a14-430e-9f8f-f883bb37fd5a)

The temperature hyperparameters for each loss function (**cosine_tau**, **ibn_tau** and **angle_tau**) can be modified in the **loss_kwargs** along with their individual weights:

![Loss_kwargs](https://github.com/rahulsikder223/fine-tuning-llm/assets/26866342/797476c8-010e-4162-8cbc-9512ae7200fc)

The results matrix can be exported into .npy files once the process is complete, which can be fed to the **results_plots_senteval_sts.ipynb** file for plotting the results. This gives an insight about the overall model performances across all datasets using all possible loss function combinations.
