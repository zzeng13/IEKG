# IEKG  
  
Data and knowledge model implementation for IEKG -- the **I**diomatic **E**xpression **K**nowledge **G**raph. 
  
  
## Description  
  
This repo contains the data and knowledge modeling implementation for the Idiomatic Expression Knowledge Graph from the paper: *IEKG: A Commonsense Knowledge Graph for Idiomatic Expressions*.  
  
The paper is published in EMNLP 2023. Please refer to the [paper](https://aclanthology.org/2023.emnlp-main.881/) for more details.  
  
The formatted data for IEKG is in: `./iekg_data/iekg_knowledge_tuples_processed.jsonl` . The data file contains a list of directories in which each data instance is a knowledge tuple in the following format: 
```
{
        "idiomatic_event": "PersonX wins by a hair's breadth",
        "relation_type": "xAttr",
        "annotation": "Ambitious",
        "idiom": "a hair's breadth",
        "defn": "A very small amount or short length"
}
```
where `idiomatic_event`, `relation_type`, and `annotation` are the head, relation, and tail annotation of the knowledge tuple and `idiom` contains the original idiom, and `defn` contains its definition as supplied by the annotators.

The implementation of the knowledge model training is under the direction `./code_km`. Please see the `README.md` under the directory for more details. 

## Acknowledgement

We acknowledge and thank the assistance of students from Washington University in St. Louis: Razi Ahmed Khan, Andrew Zhang, and Zachary Kuo, in creating and curating the knowledge graph. This research was supported in part by the National Science Foundation under Grant No. IIS 2230817 and by a U.S. National Science Foundation and Institute of Education Sciences grant (2229612). 


## Authors  
   
Ziheng Zeng (zzeng13@illinois.edu)  
  
## Version History  
  
* 0.2  
    * Added data for training knowledge model. 
* 0.1  
  * Initial Release  
  
## License  
  
This project is licensed under the MIT License.
