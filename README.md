## Install

I did not keep track of what needed to be installed for this to function, I can go back and do this later.

## Models

To select a model, change the "modelName" hyperparameter in the second cell.

### OpenAI

For this model to work, one simply needs to aa a text file containing their OpenAI key named gptKey.txt.

### BART, Llama

This should work out of the box after making the requisite installs.
(NOTE: when using Llama, must run python executable with torchrun)

### Flan

This model does not currently work because of its retrictive token limits and because it is trained for short answers.
