### BERT-ASC: Auxiliary-Sentence Construction for Implicit Aspect Learning in Sentiment Analysis [[1]](url). 

to intitialize BERT-ASC by PT model, kindly download the PT from  [here](https://drive.google.com/file/d/11pceo04PfR6W75DPCPBgZIdBxG6HP8RR/view?usp=sharing)


##### train BERT-ASC: 
	python code/run.py --dataset semeval --device cuda:0
	
##### evaluate BERT-ASC: 
	python code/evaluate.py --dataset semeval --device cuda:0
  
  
### Processing:
##### generate seed words for a given dataset (e.g., semeval): 
	python L-LDA/run_l_LDA.py --dataset semeval
note that the code of L-LDA is available [here](https://github.com/JoeZJH/Labeled-LDA-Python) 
##### generate  the semantic candidates: 
	python ASC_generating/semantic_candidate_generating.py --dataset semeval

##### capture the synticatic ifnormation: 
	python ASC_generating/opinion_words_extracting.py --dataset semeval
