import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from utils import match_seq_len
from utils import collate_fn
from utils import reset_weights

# set the dataset path
DATASET_DIR = data/"

class processingData(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR, for_what = "train") -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        if "train" in for_what:
            self.dataset_path = os.path.join(
                self.dataset_dir, for_what+".csv"
            )
        else:
            self.dataset_path = os.path.join(
                self.dataset_dir, "test.csv"
            )

        #if pkl files already exist, load them to the self's objects. 
        #Otherwise, read the data and preprocess() the data and put the output to the objects
        if os.path.exists(os.path.join(self.dataset_dir, for_what+"_"+ "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, for_what+"_"+ "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir,for_what+"_"+ "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir,for_what+"_"+ "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, for_what+"_"+"u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir,for_what+"_"+ "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir,for_what+"_"+ "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx = self.preprocess(for_what=for_what)
            
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self, for_what):
        df = pd.read_csv(self.dataset_path, encoding="ISO-8859-1")
        df = df[(df["Label"] == 0).values + (df["Label"] == 1).values]

        u_list = np.unique(df["SubjectID"].values)
        q_list = np.unique(df["ProblemID"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["SubjectID"] == u].sort_values("AttemptID")

            q_seq = np.array([q2idx[q] for q in df_u["ProblemID"].values])
            r_seq = df_u["Label"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        with open(os.path.join(self.dataset_dir,for_what+"_"+ "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, for_what+"_"+"r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir,for_what+"_"+ "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, for_what+"_"+"u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir,for_what+"_"+ "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, for_what+"_"+"u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx