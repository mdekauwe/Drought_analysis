#!/usr/bin/env python
"""
Summarise the params into PFTs
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (08.08.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

fname = "/Users/mdekauwe/Dropbox/ARC drought Trial/Parameters for Martin/parameters/parameters.csv"
df = pd.read_csv(fname)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

rf = df[df.spp == "Asm"].mean(axis=0)
wsf = df[(df.spp == "Egr") | (df.spp == "Evi")].mean(axis=0)
dsf = df[(df.spp == "Aco") | (df.spp == "Cgu") | (df.spp == "Esi")].mean(axis=0)
grw = df[(df.spp == "Ebl") | (df.spp == "Ema") | (df.spp == "Eme")].mean(axis=0)
saw = df[(df.spp == "Aan") | (df.spp == "Ela") | (df.spp == "Epo")].mean(axis=0)

sites = pd.concat([rf, wsf, dsf, grw, saw], axis=1)
sites.columns = ["rf","wsf","dsf","grw","saw"]

# Till its fixed, gap fill the rf Kplant
sites["rf"]["Kplant"] = df.Kplant.mean()
sites = sites.drop(["Cs", "Cl"])
print(sites)
