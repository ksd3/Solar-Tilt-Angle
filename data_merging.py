import pandas as pd
from glob import glob
stock_files=sorted(glob('*.csv'))
df=pd.concat((pd.read_csv(file).assign(filename=file) for file in stock_files),ignore_index=True);
df.to_csv(r'test.csv',index=False)
