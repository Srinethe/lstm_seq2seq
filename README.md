# lstm_seq2seq

   There are multiple ways to approach sequence to sequence mapping , in which we proceeded with the RNN(Recurrent Neural Network) approach. We considered two cases while applying sequence to sequence mapping:

1)  When input and output sequences have same length.
2)  When input and output sequences have different length. 
      
   Since the standard and the unstandard description have different string length in our dataset we proceeded with the case two.

•	A RNN layer acts as "encoder". It processes the input sequence and returns its own internal state. This state will serve as the "context", or "conditioning", of the decoder in the next step.

•	Another RNN layer acts as "decoder". It is trained to predict the next characters of the target sequence, given previous characters of the target sequence. 

•	Specifically, it is trained to turn the target sequences into the same sequences but offset by one timestep in the future, a training process called "Teacher forcing". The encoder uses as initial state the state vectors from the encoder, which is how the decoder obtains information about what it is supposed to generate. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.
 
In inference mode, i.e. when we want to decode unknown input sequences, we go through a slightly different process:

•	Encode the input sequence into state vectors.

•	Start with a target sequence of size 1 (just the start-of-sequence character).

•	Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character.

•	Sample the next character using these predictions (we simply use argmax).

•	Append the sampled character to the target sequence

•	Repeat until we generate the end-of-sequence character or we hit the character limit.

  Based on the above steps , we implemented for the populated data set taking  first 120 samples for the training set and next 30 for the test set which is the usual 80:20 ratio for training a model. We trained the model with a batch size of 30 and number of epoches as 1000. After training the model, the code displays the corresponding standard description for the first 100 unstandard descriptions.This would take an average of 15 minutes as the number of epoches is 1000. 


OUTPUT OF THE CODE :

Input sentence: Eng Oil Pres Ab
Decoded sentence: Engine Oil Pressure Left

-
Input sentence: Eng Oil Pres L
Decoded sentence: Engine Oil Pressure Left

-
Input sentence: Eng Oil Pres R
Decoded sentence: Engine Oil Pressure Right
