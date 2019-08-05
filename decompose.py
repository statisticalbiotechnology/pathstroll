#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.DataFrame()
for r, d, f in os.walk("./luad_tcga_pub"):
    for fname in f:
        frame = pd.read_csv(os.path.join(r, fname), delimiter='\t')
        frame.drop(columns=['#Hugo_symbol'], inplace = True)
        # Xr = frame.to_numpy()
        Xr = frame.values
        X = Xr - Xr.mean(axis=1, keepdims=True)
        svd = TruncatedSVD(n_components=1)
        svd.fit(X.T)
        V = svd.transform(X.T)
        # print(V.shape)
        model = GaussianMixture(n_components=1,covariance_type="spherical").fit(V)
        sc = model.score(V)
        print(fname, sc)
        if sc<-3.4:
            sns.distplot(V, rug=True)
            plt.show()
