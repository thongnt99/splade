import argparse 
import logging 
import numpy as np 
import os 
#############################################################################################
#####################################PARSE ARGUMENTS#########################################
#############################################################################################
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", type=str, default=None, help="path to corpus")
parser.add_argument("--query_path", type=str, default=None, help="path to query collection")
args = parser.parse_args()

collections = []

#############################################################################################
#####################################READING DATAS#########################################
#############################################################################################
logging.info("Reading {}".format(args.corpus_path))
with open(args.corpus_path, "r", encoding="utf-8") as fp:
    for line in fp:
        _, text = line.strip().split("\t")
        collections.append(text)


logging.info("Reading {}".format(args.query_path))
with open(args.query_path, "r", encoding="utf-8") as fp:
    for line in fp:
        _, text = line.strip().split("\t")
        collections.append(text)

# shuffle data 
indices = list(range(len(collections)))
indices = np.random.shuffle(indices)

# spliting data
n_train = int(len(collections)*0.7)

if not os.path.exists("msmarco-data"):
    logging.info("Making output folder")
    os.mkdir("msmarco-data")
with open("msmarco-data/train.txt","w") as f_train:
    for idx in indices[:n_train]:
        f_train.write(collections[idx]+"\n")

with open("msmarco-data/eval.txt","w") as f_train:
    for idx in indices[n_train:]:
        f_train.write(collections[idx]+"\n")
