#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 12:03:15 2026

@author: songyangwang
"""

import pandas as pd
import pyarrow

data = pd.read_parquet("data/hard/track2_data.parquet")
print(data.shape)  # (n_samples, 1024)