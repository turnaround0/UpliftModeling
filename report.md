# Uplift implementation report
Deadline: 2019/08/29

Dataset
*** Dataset name: hillstrom
CN    19044
CR     2262
TN    18149
TR     3238
3238 / 18149 - 2262 / 19044 = 0.0596

criteo: imbalanced dataset
CN    539
CR     78
TN    682
TR     92
92 / 682 - 78 / 539 = -0.0098

*** Dataset name: lalonde
T
0.0    5090.048302
1.0    5976.352033

## 1. Figure 5. Results of the wrapper variable selection procedure (general wrapper approach)
Figure 5 is skipped for criteo dataset.

Variable selection has been applied in combination with tma, dta, pes, and trans approaches,
but not in combination with tree-based approaches since these internally incorporate a variable selection procedure.
So there is no need from a practical perspective, and in addition,
the performance of tree-based approaches may deteriorate when adding an external variable selection procedure.

As can be seen from the experimental results shown in Figure 5, generally, the best performance in terms of the
Qini measure is achieved with a relatively small number of predictor variables in the model, that is, in between 5
and 15 predictor variables.

When adding predictor variables to the model, at first the performance of the model improves. When the optimal
number of predictor variables resulting in maximum performance is reached and more predictor variables are
added to the model, a downward trend in performance is observed. An explanation of the downward trend is
that the effect of the campaign is captured or described by a limited number of predictor variables. When adding
more predictor variables, which are either only weakly related to the effect of the campaign or strongly correlated
with the predictors already in the model, in fact noise is added to the data, leading to a decrease in terms of the
performance and generalization power of the models, or an increase in terms of stability or overfitting.


## 2. Table 6. Qini values of all methods

generally, the reported Qini > Qini Top 30% > Qini Top 10%.
The area under the Qini curves can be expected to be larger for an increasing fraction of the population being selected.

The standard deviation: This may signal an issue with the stability of the models
At first sight, it appears that uplift random forests consistently performs among the best techniques.
Two more techniques, dta and tma, as well perform consistently well across multiple data sets.

Uplift random forests generally performs well on most data sets.
Also, lai, glai, tma, and dta perform well in comparison.
Surprisingly, the tma technique (i.e., the two-model approach),
which is often referred to as the naive approach since relatively simple in terms of setup,
appears nonetheless to perform well in specific cases.


First of all, the Qini measures only provide an indication how well a model is performing when
compared with other models on the same data set. A comparison across data sets is not supported,
since these measures are not normalized and therefore depend on characteristics of the application.

in addition, the Qini metrics evaluate how well an uplift model ranks the full population.


## 3. Figure 7. Qini curve for methods (average of Qini curves for all folds)



## 4. Figure 8. Qini curve for each folds of methods: methods having best and worst Qini value for each data set

## 5. Figure 9. performance a of all methods on Fold 1 & Fold 3
