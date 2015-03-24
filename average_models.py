import numpy as np
import pandas as pd

dbn = pd.read_csv('Preds/dbn.csv')
dbn = dbn.drop('id', axis=1)

xgboost = pd.read_csv('Preds/xgboost.csv')
xgboost = xgboost.drop('id', axis=1)

graphlab = pd.read_csv('Preds/graphlab_submission.csv')
graphlab = graphlab.drop('id', axis=1)

mlp = pd.read_csv('Preds/kayak_logistic_mlp_preds.csv')
mlp = mlp.drop('id', axis=1)

sample = pd.read_csv('Data/sampleSubmission.csv')

avg = dbn.values + xgboost.values + graphlab.values + mlp.values / 4.0

# ----------------------  create submission file  -----------------------------
avg = pd.DataFrame(avg, index=sample.id.values, columns=sample.columns[1:])
avg.to_csv('Preds/avg_dbn_xgboost_gl_mlp.csv', index_label='id')
