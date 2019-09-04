#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

print("## Analysis of LUAD pathways")
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
        out_frame = pd.DataFrame(data=V.T,columns=frame.columns,index=[0])
        out_frame.sort_values(by=[0], axis=1, ascending=True, inplace=True)
        dataname = os.path.join( "./out/", os.path.splitext(fname)[0]+'.txt')
        out_frame.to_csv(path_or_buf=dataname, sep='\t',index=False)
        figname = os.path.join( "./img/", os.path.splitext(fname)[0]+'.png')
        # Write markdown links
        print('![plt]({}) [values]({})'.format(figname,dataname))
        sns.set_style("ticks")
        sns.distplot(V, rug=True)
        plt.savefig(figname)
        plt.clf()
