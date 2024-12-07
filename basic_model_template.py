from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


class InitialModel:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median', add_indicator=True)
        self.oh_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.stat_model = sm.OLS

    def merge_categories(self, df):
        mapping_dict = {}
        values_not_to_change = {}

        def _map_logic(val, maps, dont_change_list):
            if val in dont_change_list:
                return val
            if val in maps:
                return maps[val]
            else:
                return maps['__other__']

        for c, maps in mapping_dict.items():
            df.loc[:, c] = df[c].apply(lambda x: _map_logic(x, maps, values_not_to_change.get(c, [])))
        return df

    def make_dm(self, df, train_mode=True):
        l = []
        X_num = ...
        X_cat = ...
        if train_mode:
            X_num = ...
        else:
            X_num = ...
        l.append(...)
        return pd.concat(l, axis=1)



def basic_model_workflow(df_train, df_test, y_train, y_test):
    m = InitialModel()
    dm_train = m.make_dm(df_train)
    ols = sm.OLS(np.log(y_train), sm.add_constant(dm_train))
    res = ols.fit()
    plt.scatter(res.predict(), np.log(y_train), marker='.', alpha=.1)
    plt.axline((10, 10), slope=1, linestyle='--')
    plt.show()
    res.summary()




