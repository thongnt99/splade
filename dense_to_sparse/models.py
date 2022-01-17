#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License APACHE2

from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor

class MeanPooling(nn.Module):
    def __init__(self, embedding_dimension: int):
        super(MeanPooling, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.config_keys = ["embedding_dimension"]
    
    def __repr_(self):
        return "Mean Pooling ({})"
    
    def get_pooling_mode_str(self) -> str: 
        return "Mean Pooling"
    
    def forward(self, features: Dict[str, Tensor]):    
        last_hidden_states = features["last_hidden_states"] # first dim: batch, second dim: seq len, third dim: vector length
        attention_mask = features["attention_mask"].unsqueeze(-1)
        seq_lens = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_embedding = torch.sum(last_hidden_states*attention_mask, dim=1)/seq_lens
        features.update({"dense_embedding": mean_embedding})
        return features
    
    def get_embedding_dimension(self):
        return self.embedding_dimension
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return MeanPooling(**config) 

class Dense2Sparse(nn.Module):
    def __init__(self, model_type="1-layer", out_dim=30522):
        super(Dense2Sparse, self).__init__()
        if model_type == "1-layer":
            self.transfer_model = nn.Sequential(
                nn.Linear(768, out_dim)
            )
        elif model_type == "2-layer":
            self.transfer_model = nn.Sequential(
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.Linear(768, out_dim)
            )
        elif model_type == "mlm":
            transformer = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.transfer_model = nn.Sequential(
                transformer.vocab_transform,
                transformer.vocab_layer_norm,
                transformer.vocab_projector
            )
        elif model_type == "mlm-head":
            transformer = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.transfer_model = nn.Sequential(
                transformer.vocab_projector
            )
        else:
            raise ValueError("model_type {} is not valid. Select: 1-layer, 2-layer, mlm".format(model_type))

    def forward(self, batch_rep):
        return self.transfer_model(batch_rep)

class Dense2SparseModel(nn.Module):
    """
    Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param dense_model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param model_type: 1-layer, 2-layer, mlm head
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, dense_model_name_or_path: str, model_type: str = "1-layer", max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False, use_log: bool = False):
        super(Dense2SparseModel, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        self.use_log = use_log

        # dense model
        dense_config = AutoConfig.from_pretrained(dense_model_name_or_path, **model_args, cache_dir=cache_dir)
        dense_model = AutoModel.from_pretrained(dense_model_name_or_path, config=dense_config, cache_dir=cache_dir)
        for param in dense_model.parameters():
            param.requires_grad = False
        self.dense_model = torch.nn.DataParallel(dense_model)
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        self.mean_pooling = torch.nn.DataParallel(MeanPooling(768)) 
        self.dense_to_sparse_query = torch.nn.DataParallel(Dense2Sparse(model_type=model_type, out_dim=self.get_word_embedding_dimension()))
        self.dense_to_sparse_doc = torch.nn.DataParallel(Dense2Sparse(model_type=model_type, out_dim=self.get_word_embedding_dimension()))
        if model_type == "mlm":
            for param in self.dense_to_sparse_doc.module.transfer_model[-1].parameters():
                param.requires_grad = False 
            for param in self.dense_to_sparse_query.module.transfer_model[-1].parameters():
                param.requires_grad = False 

        self.max_seq_length = max_seq_length

    def __repr__(self):
        return "Dense2Sparse ({}) with Transformer model: {}".format(self.get_config_dict(), self.dense_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
        # dense representation 
        last_hidden_state = self.dense_model(**trans_features).last_hidden_state
        features.update({'last_hidden_states': last_hidden_state})
        features = self.mean_pooling(features)
        
        # covert dense to sparse 
        if features["type"] == "query":
            sparse_from_dense = self.dense_to_sparse_query(features["dense_embedding"])
        elif features["type"] == "doc":
            sparse_from_dense = self.dense_to_sparse_doc(features["dense_embedding"])


        # use relu and log to enforce some sparsity     
        if self.use_log:
            sparse_from_dense = torch.log(1 + torch.relu(sparse_from_dense))
        else:
            sparse_from_dense = torch.relu(sparse_from_dense)

        features.update({"sparse_from_dense": sparse_from_dense})
        return features

    def get_word_embedding_dimension(self) -> int:
            return self.dense_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.dense_tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        # self.auto_model.module.save_pretrained(output_path)
        # self.tokenizer.save_pretrained(output_path)
        torch.save(self.dense_to_sparse_doc.module.state_dict(), f"{output_path}/dense_to_sparse_doc.pt")
        torch.save(self.dense_to_sparse_query.module.state_dict(), f"{output_path}/dense_to_sparse_query.pt")
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Dense2SparseModel(model_name_or_path=input_path, **config)