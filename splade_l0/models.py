#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License APACHE2

from torch import nn
import torch.nn.functional as F
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
        features.update({"mean_dense_embedding": mean_embedding})
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

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]        
        grad_output[input>=1] = 0 
        grad_output[input<=0] = 0
        return grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        # self.ste = StraightThroughEstimator()
        self.config_keys = ["word_embedding_dimension"]

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        ## Pooling strategy
        sentence_embedding, _ = torch.max(torch.relu(token_embeddings) * attention_mask.unsqueeze(-1), dim=1)
        # l_0 = self.ste(sentence_embedding)
        l_0 = torch.nn.functional.tanh(sentence_embedding)
        features.update({'sentence_embedding': sentence_embedding, "l_0": l_0})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)

class MLMTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None, scratch: bool = False, freeze_vocab = False):
        super(MLMTransformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        if freeze_vocab is True:
            # freeze the vocabulary projecion layer to ensure the zipfian distribution of the output 
            for param in model.vocab_projector.parameters():
                param.requires_grad = False 
                
        self.auto_model = torch.nn.DataParallel(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        self.pooling = torch.nn.DataParallel(Splade_Pooling(self.get_word_embedding_dimension())) 
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length
        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "MLMTransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=True, output_hidden_states=True)
        output_tokens = output_states.logits
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})
        features = self.pooling(features)
        return features

    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
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

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

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
        return MLMTransformer(model_name_or_path=input_path, **config)