import pandas as pd
import numpy as np
import seaborn as sns; sns.set(); sns.set_style('darkgrid'); sns.set_palette('RdBu')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


train_df = pd.read_csv('PM_train.txt', sep=' ',header=None)
col_names = ['bestofthebest','best1','best2','best3','worst','s1','s2','s3','s4','s5','s6','s7','s8',\
             's9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21', 's22', 's23']
train_df.columns = col_names
labels = pd.read_csv('PM_truth.txt', sep=' ',header=None); print(labels.shape)
label_name = ['lifetime'][0] # small fixme
labels = labels.rename(columns={0: label_name})
labels['bestofthebest'] = labels.index + 1
labels.drop(1, axis=1, inplace=True)


features = [f for f in train_df.columns]
N = int(train_df.columns.shape[0])

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

'''doubtful heuristics - pay attention on ids where feature value >= 90 quantile
                            FIXME'''
box_data = [{'y': np.array(labels['lifetime'])[train_df.query(f'{fre} > {fre}.quantile(q=0.9)').bestofthebest.unique() - 1], 
    'type':'box', 'marker':{'color': c[features.index(fre)]},
    'name': fre} for fre in features]

layout = go.Layout(
    title='Супер важный бизнес график возможных доходов со сложной статистикой',
    yaxis = {'title': 'Lifetime'}
)   

fig = go.Figure(data = box_data, layout = layout)
iplot(fig)
