
 # BERT-ASC 
 A model for implicit aspect sentiment analysis.
 
 This is the source code for the paper: Murtadha, Ahmed, et al. "BERT-ASC: Auxiliary-Sentence Construction for Implicit Aspect Learning in Sentiment Analysis" [[1]](https://arxiv.org/abs/2203.11702). 
 

 

# Data



The datasets used in our experminents can be downloaded from this [SemEval](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools). 

# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# Pre-processing

* Generate seed words for a given dataset (e.g., semeval): 
	* Go to L-LDA/  
	* Run the following code
	```
	python run_l_LDA.py --dataset semeval
	```
The original code of L-LDA is publicly [available](https://github.com/JoeZJH/Labeled-LDA-Python) 
* Generate the semantic candidates: 
	* Go to  ASC_generating/  
	* Run the following code to extract the semantic candidates
	```
	python semantic_candidate_generating.py --dataset semeval
	```
* Generate the synticatic candidates: 
	* Run the following code to generate the synticatic informatiom
	```
	python ASC_generating/opinion_words_extracting.py --dataset semeval
	```



# Training: 
* To train  BERT-ASC: 
	* Go to  code/  
	* Run the following code 
	```
	python code/run.py --dataset semeval --device cuda:0
	```
	* The params could be :
		- --dataset =\{semeval,sentihood\}	
		- --device ="cuda:0"
	
# Evalutaion: 	
* To evaluate  BERT-ASC:
	* Go to  code/  
	* Run the following code 
	```
	python code/evaluate.py --dataset semeval --device cuda:0
	```
	* The params could be :
		- --dataset =\{semeval,sentihood\}	
		- --device ="cuda:0"	

 If you use the code,  please cite the paper: 
```
@article{murtadha2022bert,
  title={BERT-ASC: Auxiliary-Sentence Construction for Implicit Aspect Learning in Sentiment Analysis},
  author={Murtadha, Ahmed and Pan, Shengfeng and Wen, Bo and Su, Jianlin and Zhang, Wenze and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2203.11702},
  year={2022}
}
```
