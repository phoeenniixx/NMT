# **Neural Machine Translation**

1. Neural Machine Translation (NMT) is the task of using artificial neural network models for translation from one language to the other.
2. The NMT model generally consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation.
3. This problem can be thought as a prediction problem, where given a sequence of words in source language as input, task is to predict the output sequence of words in target language.
4. The dataset comes from http://www.manythings.org/anki/, where you may find tab delimited bilingual sentence pairs in different files based on the source and target language of your choice


The model is trained to learn the mapping between the source language and target language, allowing it to generate translations for new input sentences. The training process involves optimizing the model's parameters to minimize the difference between predicted and actual translations. The effectiveness of the model can be evaluated on the test set, and adjustments can be made to improve its performance if necessary.

The project involves building a machine translation model using a sequence-to-sequence (seq2seq) architecture with an encoder-decoder framework. The goal is to translate sentences from one language to another. The project can be broken down into several key steps:

Step 1: Download and Clean the Data
1. Download a dataset containing language pairs (source and target phrases).
2. Extract the data and read it into a format suitable for processing.
3. Clean the data by removing non-printable characters, punctuation, and non-alphabetic characters.
4. Convert the text to lowercase for uniformity.

Step 2: Split and Prepare the Data for Training
1. Split the cleaned data into training and testing sets.
2. Create separate tokenizers for the source and target languages to convert text into numerical sequences.
3. Encode and pad the input sequences (source language) and output sequences (target language) based on the individual tokenizers and maximum sequence lengths.
4. Perform one-hot encoding on the output sequences, as the goal is to predict words in the target language.
   
Step 3: Define and Train the RNN-based Encoder-Decoder Model
1. Define a sequential model with two main parts: Encoder and Decoder.
2. In the Encoder, pass the input sequence through an Embedding layer to train word embeddings for the source language. Additionally, use one or more RNN/LSTM layers to capture sequential information.
3. Connect the Encoder to the Decoder using a Repeat Vector layer to match the shapes of the output from the Encoder and the expected input by the Decoder.
4. In the Decoder, stack one or more RNN/LSTM layers and add a Time Distributed Dense layer to produce outputs for each timestep.

   

Evaluating the accuracy of LSTM-based neural machine translation (NMT) models is essential to gauge their performance in generating meaningful translations. BLEU (Bilingual Evaluation Understudy) score is a widely-used metric for this purpose, providing a quantitative measure of the similarity between the model's output and human-generated references.

For the specific NMT model in question, the obtained BLEU scores are as follows:

      BLEU-1: 0.617742
      BLEU-2: 0.407798
      BLEU-3: 0.315834
      BLEU-4: 0.188856
These scores reflect the model's ability to produce translations that align with the ground truth. While higher BLEU scores generally indicate better translation accuracy, it's crucial to consider the specific characteristics of the dataset and the translation task.

It's worth noting that the model's performance could potentially be enhanced by training on a larger dataset. However, due to constraints on machine resources, a smaller dataset was used, consisting of only 20,000 lines out of the original 229,803 lines. Despite the limited dataset size, the provided BLEU scores offer valuable insights into the model's translation quality given the available resources.

Choosing a smaller dataset is a pragmatic decision influenced by machine-specific limitations. Even with these constraints, the model has demonstrated respectable performance, and future improvements may be explored by addressing these limitations and working with a more extensive training dataset.




