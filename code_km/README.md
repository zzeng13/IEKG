
  
# IEKG  
  
Data and knowledge model implementation for IEKG -- the **I**diomatic **E**xpression **K**nowledge **G**raph. 
  
  
## Description  
  
This repo contains the data and knowledge modeling implementation for the Idiomatic Expression Knowledge Graph from the paper: *IEKG: Idiomatic Expression Knowledge Graph*.  
  
The paper is published in EMNLP2023. Please refer to the paper for more details.  
  
## Getting Started    
  
### Dependencies  
  
We recommend running the model training and evaluation in a Python virtual environment.  
The dependencies and required packages are listed in `requirements.txt`.  

Note that to use or fine-tune the original Atomic-2020 Knowlege model, you would need to download the checkpoint from its 
official host and set `PATH_TO_COMET_BART` in `config.py` to its proper local location.
  
  
### Data  
The main IEKG data files under `./data/` here are processed for knowledge model training and split into 
relation type split and idiom type split as described in the paper. Specifically: 
* `./data/iekg_knowledge_tuples_processed_idiom_type_split.json` contains the IEKG tuples by idiom type split;
* `./data/iekg_knowledge_tuples_processed_relation_type_split.json` contains the IEKG tuples by relation type split;

The file `atomic_additional_tokens_list.json` contains the additional tokens needed for the knowledge model's tokenizer. 


## Train Knowledge Model

To train the knowledge model: 
1. Set the configuration in `config.py`
   * Use `model_type` to select the backbone model.
   * Use `split_method` to select which data split is used for training
   * For the other settings, please follow the paper or comments in `config.py`
2. Run `train.py`

  
## Authors  
   
Ziheng Zeng (zzeng13@illinois.edu)  
  
## Version History  
  
* 0.2  
    * Added data for training knowledge model 
* 0.1  
  * Initial Release  
  
## License  
  
This project is licensed under the MIT License.