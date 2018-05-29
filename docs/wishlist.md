# General
- want to be able to fit by batch by either providing a numpy style array (so everything fits into memory) or provide a generator function
- which framework is used should be decided based on what is in the environment / based on a user configuration if both frameworks are available
# Top level
- invisible to the user which NN framework is used
- have Models `TAE` and `VAMPNet` which can be "trained" (layer sizes, dropout, batch size, learning rate, activation functions etc)
- Trained models can be `fit`-ted on data 
# Mid level 
- abstraction layer between the actual NN-framework implementation and the top layer
# Low level
- specialization toward pytorch / TF as implementation of the abstraction layer
  - smaller dispatch interfaces separated through namespaces, e.g.,
  ```python
  deeptime.scores.tf.vamp
  deeptime.scores.torch.vamp
  ```