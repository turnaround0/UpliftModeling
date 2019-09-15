# Uplift implementation report

## 1. Check dataset
#### Dataset #1. hillstrom
Shape: (42693, 20)<br>
== Count of each group ==<br>
CN    19044<br>
CR     2262<br>
TN    18149<br>
TR     3238<br>
Uplift: 0.045233106587052985

There are enough data to train.

#### Dataset #2. lalonde
Shape: (722, 9)<br>
== Sum of each group ==<br>
T    2.163271e+06<br>
C    1.774977e+06<br>
== Count of each group ==<br>
T    425<br>
C    297<br>
== Average of each group ==<br>
T    5090.048302<br>
C    5976.352033<br>
Uplift: 886.3037307037466

Too small data prevents to train.

#### Dataset #3. criteo<br>
Shape: (1391, 3570)<br>
== Count of each group ==<br>
CN    539<br>
CR     78<br>
TN    682<br>
TR     92<br>
Uplift: -0.007555103254473797

There are too small data with many columns.<br>
Number of columns will be reduced by NIV(Net Information Value).<br>

## 2. Figure 5. Results of the wrapper variable selection procedure (general wrapper approach)
### 2.1. hillstrom
![hillstrom_Fig5](pictures/hillstrom_Fig5.png)


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

### 2.2. lalonde
![lalonde_Fig5](pictures/lalonde_Fig5.png)

Drop variables of first fold for tma are below by general wrapper approach.<br>
['hispanic', 'nodegree', 'education', 'black']<br>
These columns will be removed when training.



### 2.3. criteo
Figure 5 is skipped for criteo dataset.<br>
NIV selects 50 features among 3570 features.

50 selected variables of first fold for tma are below.<br>
['6_3', '13_0', '15_0', '17_2', '4_1', '10_1', '25_106', '1', '4_16', '4_35', '10_16', '6_4', '2', '8_252', '9_218', '10_2', '5_1458', '17_1', '4_6', '24_11', '5_5', '25_4', '8_53', '22_8', '10_29', '5_8', '4_34', '17_4', '5_2', '5_133', '25_14', '11_0', '16_2', '4_44', '6_6', '4_185', '7_3', '5_7', '25_746', '19_0', '21_23', '9_59', '6_5', '10_36', '22_28', '8_193', '6_1', '10_0', '35_145', '25_424']



## 3. Table 6. Qini values of all methods
### 3.1. hillstrom
* Dataset: hillstrom
                        Qini       Qini top 30%       Qini Top 10%
tma        0.70211 (0.16531)  0.11105 (0.03889)  0.01034 (0.01406)
dta        0.75075 (0.23852)  0.11698 (0.06860)  0.01669 (0.00776)
lai        0.76699 (0.16040)  0.14167 (0.06064)  0.01328 (0.01175)
glai       0.73533 (0.16786)  0.10444 (0.04276)  0.00818 (0.00283)
trans      0.73535 (0.23452)  0.13978 (0.06988)  0.01193 (0.01066)
urf_ed     0.67444 (0.26340)  0.09647 (0.07378)  0.01227 (0.01146)
urf_kl     0.70980 (0.17773)  0.11974 (0.03793)  0.01045 (0.00648)
urf_chisq  0.69935 (0.15433)  0.11531 (0.04215)  0.00929 (0.01029)
urf_int    0.66621 (0.27896)  0.10072 (0.07974)  0.00305 (0.01697)
Best model: lai
Worst model: urf_int


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

### 3.2. lalonde
                            Qini          Qini top 30%         Qini Top 10%
tma         35.14079 (310.46356)  28.67711 (119.94554)   7.75662 (22.42195)
dta         35.14079 (310.46356)  28.67711 (119.94554)   7.75662 (22.42195)
urf_ed      88.54089 (150.20571)  -32.61541 (51.64044)   0.15791 (12.51625)
urf_kl      73.28015 (226.82092)  -13.20159 (98.77903)  -1.77061 (20.09244)
urf_chisq  110.63669 (231.96557)   10.95601 (75.56024)   3.21292 (23.79192)
urf_int     53.16886 (119.48879)    2.48863 (54.72581)  -4.81489 (13.23914)
Best model: urf_chisq
Worst model: tma

Enlarge dataset to 10 times
* Dataset: lalonde
                      Qini          Qini top 30%        Qini Top 10%
tma  404.28101 (121.34154)  109.99043 (39.24576)  15.19122 (8.65505)


### 3.3. criteo
                         Qini        Qini top 30%        Qini Top 10%
tma        -1.67587 (1.02645)  -0.51779 (0.27925)  -0.07282 (0.04639)
dta        -1.82369 (0.72182)  -0.45862 (0.16259)  -0.07317 (0.04294)
lai        -0.62102 (1.40271)  -0.14827 (0.87637)   0.02869 (0.20201)
glai       -0.46142 (1.48412)  -0.35412 (0.32114)  -0.03212 (0.07118)
trans      -0.61331 (1.39994)  -0.14827 (0.87637)   0.02869 (0.20201)
urf_ed     -1.54690 (1.32393)  -0.23615 (0.34802)  -0.01105 (0.05361)
urf_kl     -0.69568 (1.24608)  -0.08373 (0.43423)  -0.00500 (0.05097)
urf_chisq  -0.37198 (1.04248)   0.00028 (0.33598)   0.01297 (0.05628)
urf_int    -1.28974 (1.34804)  -0.16746 (0.37185)  -0.03199 (0.03871)

Best model: urf_chisq
Worst model: dta



## 4. Figure 7. Qini curve for methods (average of Qini curves for all folds)
### 4.1. hillstrom
![hillstrom_Fig7](pictures/hillstrom_Fig7.png)

### 4.2. lalonde
![lalonde_Fig7](pictures/lalonde_Fig7.png)

### 4.3. criteo
![criteo_Fig7](pictures/criteo_Fig7.png)


## 5. Figure 8. Qini curve for each folds of methods: methods having best and worst Qini value for each data set
### 5.1. hillstrom
![hillstrom_Fig8](pictures/hillstrom_Fig8.png)

### 5.2. lalonde
![lalonde_Fig8](pictures/lalonde_Fig8.png)

### 5.3. criteo
![criteo_Fig8](pictures/criteo_Fig8.png)

## 6. Figure 9. performance a of all methods on Fold 1 & Fold 3
### 6.1. hillstrom
![hillstrom_Fig9](pictures/hillstrom_Fig9.png)

### 6.2. lalonde
![lalonde_Fig9](pictures/lalonde_Fig9.png)

### 6.3. criteo
![criteo_Fig9](pictures/criteo_Fig9.png)


## 7. Test result (temp)
                        Qini       Qini top 30%       Qini Top 10%
dt_ed      0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_ext  0.45372 (0.22222)  0.10841 (0.06495)  0.01108 (0.01433)

                        Qini       Qini top 30%       Qini Top 10%
dt_ed      0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_ext  0.61885 (0.21881)  0.12435 (0.07068)  0.01387 (0.01108)

                         Qini        Qini top 30%        Qini Top 10%
dt_ed      -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_ext  -0.70796 (1.32248)  -0.15564 (0.26098)  -0.02876 (0.03580)

* Dataset: lalonde
                            Qini          Qini top 30%         Qini Top 10%
dt_ed       71.06822 (150.09804)  12.29716 (111.57992)  -0.35692 (20.24544)
dt_ed_ext  174.70211 (262.88161)  16.45573 (109.74249)   4.84953 (24.04970)


* Dataset: criteo
                         Qini        Qini top 30%        Qini Top 10%
dt_ed      -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_ext  -0.38810 (1.31110)  -0.15021 (0.26694)  -0.02876 (0.03580)

* Dataset: hillstrom
                           Qini        Qini top 30%       Qini Top 10%
dt_ed_focus  -0.10657 (0.23124)  -0.01634 (0.02499)  0.00036 (0.00939)


* Dataset: hillstrom
                      Qini       Qini top 30%       Qini Top 10%
tma_ext  0.69419 (0.18975)  0.06744 (0.05046)  0.00389 (0.00834)
tma      0.75468 (0.17469)  0.10917 (0.04544)  0.00804 (0.00798)




original)
dt_ed      0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)

1.5)
* Dataset: hillstrom
                           Qini       Qini top 30%        Qini Top 10%
dt_ed_focus  -0.03717 (0.18845)  0.00080 (0.07029)  -0.00055 (0.00960)

1.0)
* Dataset: hillstrom
                           Qini       Qini top 30%       Qini Top 10%
dt_ed_focus  -0.07323 (0.11295)  0.00787 (0.03161)  0.00276 (0.00497)

2.0)
* Dataset: hillstrom
                          Qini       Qini top 30%        Qini Top 10%
dt_ed_focus  0.06318 (0.14218)  0.00612 (0.05909)  -0.00300 (0.01159)


* Dataset: hillstrom
                           Qini       Qini top 30%       Qini Top 10%
dt_ed_focus   0.50798 (0.30439)  0.11340 (0.05877)  0.02010 (0.00993)
dt_ed2_focus  0.43900 (0.21439)  0.09853 (0.02409)  0.01173 (0.00421)
dt_ed3_focus  0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)

with -0.5
* Dataset: hillstrom
                          Qini       Qini top 30%       Qini Top 10%
dt_ed        0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_focus  0.13254 (0.19186)  0.03753 (0.03300)  0.00485 (0.01231)

* Dataset: lalonde
                             Qini          Qini top 30%         Qini Top 10%
dt_ed        71.06822 (150.09804)  12.29716 (111.57992)  -0.35692 (20.24544)
dt_ed_focus  172.23677 (73.51420)   65.33883 (46.37669)   10.68576 (6.23433)

* Dataset: criteo
                           Qini        Qini top 30%        Qini Top 10%
dt_ed        -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_focus  -0.05599 (0.82595)  -0.11996 (0.21418)  -0.02763 (0.03276)


with -1.0
* Dataset: hillstrom
                          Qini       Qini top 30%       Qini Top 10%
dt_ed        0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_focus  0.50798 (0.30439)  0.11340 (0.05877)  0.02010 (0.00993)

* Dataset: hillstrom (modified)
                          Qini       Qini top 30%       Qini Top 10%
dt_ed        0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_focus  0.48935 (0.30638)  0.11898 (0.05120)  0.02110 (0.00879)


* Dataset: lalonde
                              Qini          Qini top 30%         Qini Top 10%
dt_ed         71.06822 (150.09804)  12.29716 (111.57992)  -0.35692 (20.24544)
dt_ed_focus  267.24115 (257.08685)   38.73444 (92.48329)    9.55094 (2.76008)

* Dataset: lalonde (modified)
                              Qini          Qini top 30%         Qini Top 10%
dt_ed         71.06822 (150.09804)  12.29716 (111.57992)  -0.35692 (20.24544)
dt_ed_focus  231.47173 (239.53270)   43.95370 (84.63859)    8.86961 (4.00321)


* Dataset: criteo
                           Qini        Qini top 30%        Qini Top 10%
dt_ed        -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_focus  -0.01896 (0.86583)   0.02197 (0.27200)  -0.01335 (0.04281)

* Dataset: criteo (modified)
                           Qini        Qini top 30%        Qini Top 10%
dt_ed        -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_focus   0.22925 (0.77091)   0.03448 (0.23480)  -0.01632 (0.03924)


with -1.5
* Dataset: hillstrom
                          Qini       Qini top 30%       Qini Top 10%
dt_ed        0.58786 (0.18384)  0.13422 (0.05715)  0.01635 (0.01593)
dt_ed_focus  0.60363 (0.22129)  0.12457 (0.03812)  0.01480 (0.00608)

* Dataset: lalonde
                             Qini           Qini top 30%         Qini Top 10%
dt_ed        71.06822 (150.09804)   12.29716 (111.57992)  -0.35692 (20.24544)
dt_ed_focus  23.49730 (307.74133)  -34.43657 (113.76542)  -2.53345 (19.11933)

* Dataset: criteo
                           Qini        Qini top 30%        Qini Top 10%
dt_ed        -0.42581 (1.50447)  -0.05722 (0.43278)   0.00108 (0.09051)
dt_ed_focus  -0.24321 (0.97695)  -0.07522 (0.33122)  -0.01668 (0.04200)
