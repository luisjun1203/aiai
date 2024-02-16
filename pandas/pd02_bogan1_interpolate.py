import pandas as pd
from datetime import datetime
import numpy as np

dates = ['2/16/2024','2/17/2024','2/18/2024',
         '2/19/2024','2/20/2024','2/21/2024']

dates = pd.to_datetime(dates)
print(dates)
print('======================================================')
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan])
print(ts)

ts = ts.interpolate()
print(ts)
