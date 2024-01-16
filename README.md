# SRU
The implementation of the SRU [(Simple Recurrent Unit)](https://arxiv.org/abs/1709.02755) model involves training it to learn from data through a counting task. For instance, if the input sequence is $x_1 = 1$, $x_2 = 2$, $x_3 = 3$, the model can be trained to predict the next value in the sequence, such that it would forecast $x_4 = 4$

Some examples of the model output are as follows:
```
Given a sequence: x1=2, x2=3, x3=4, SRU predictd that x4=5
Given a sequence: x1=16, x2=17, x3=18, SRU predictd that x4=19
Given a sequence: x1=32, x2=33, x3=34, SRU predictd that x4=35
Given a sequence: x1=49, x2=50, x3=51, SRU predictd that x4=52
Given a sequence: x1=71, x2=72, x3=73, SRU predictd that x4=74

```
All the predictions of eval_dataset can be seen in the [eval_results.csv](https://github.com/GSYfate/SRU/blob/main/eval_results.csv).
