# Heterogeneous Syntax Fuser (HeSyFu) for SRL


Code for the ACL2021 (Finding) work [**Better Combine Them Together! Integrating Syntactic Constituency and Dependency Representations for Semantic Role Labeling**](https://aclanthology.org/2021.findings-acl.49/)





## Data

#### Donwload the SRL dataset:
* [CoNLL05](https://www.cs.upc.edu/˜srlconll/soft.html)
* [CoNLL09](https://catalog.ldc.upenn.edu/LDC2012T03)
* [CoNLL12](https://catalog.ldc.upenn.edu/LDC2013T19)


#### Ensemble the heterogeneous syntax annotations by linking the CoNLL sentences to the UPB. 

#### Format the data as CoNLLU-like, make sure the dependency syntax (head & dependent label) and the constituency syntax are will compatible to the conllu style.


#### Word embedding:
* [GloVe](https://github.com/stanfordnlp/GloVe)
* [RoBERTa (base)](https://github.com/pytorch/fairseq/tree/master/examples/RoBERTa)



## Training & Evaluating


#### Run _run_srl.py_

#### Run _run_inference.py_


***

```
@inproceedings{fei-etal-2021-better,
    title = "Better Combine Them Together! Integrating Syntactic Constituency and Dependency Representations for Semantic Role Labeling",
    author = "Fei, Hao  and
      Wu, Shengqiong  and
      Ren, Yafeng  and
      Li, Fei  and
      Ji, Donghong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "549--559",
}
