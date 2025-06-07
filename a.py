import pandas as pd
import pingouin as pg

df = pd.DataFrame({
    'targets': ['a', 'a', 'a', 'b', 'b', 'b'],
    'raters': ['r1', 'r2', 'r3', 'r1', 'r2', 'r3'],
    'ratings': [4.1, 4.3, 3.9, 2.5, 2.7, 2.6]
})

print(pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='ratings'))
