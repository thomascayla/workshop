# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
transactions_known_scored = dataiku.Dataset("transactions_known_scored")
transactions_known_scored_df = transactions_known_scored.get_dataframe()




# Write recipe outputs
data_viz = dataiku.Folder("ofSVU3Pe")
data_viz_info = data_viz.get_info()
