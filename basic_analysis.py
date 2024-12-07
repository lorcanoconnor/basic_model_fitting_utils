import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

def partial_regression_plot(df_train, dm_train, y_train, extra_feature):
    """"
    df_train: full df with extra feature to investigate.
    Note: res1 ~ 1 + res2 fits 0, beta, where beta is same coefficient as for y ~ 1 + dm + extra_feature
    (because nested orthogonal projections)
    """
    res1 = sm.OLS(y_train, sm.add_constant(dm_train), missing='drop').fit().resid
    res2 = sm.OLS(df_train[extra_feature], sm.add_constant(dm_train), missing='drop').fit().resid
    sns.scatterplot(x=res2, y=res1, marker='.', alpha=.5); plt.show()



# random forest feature importances
# --------------------------------
# from sklearn.ensemble import RandomForestRegressor as RF
# rf = RF(n_jobs=-1, n_estimators=100)
# rf.fit(..., ...)
# pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values().plot(kind='bar')




# from sklearn.inspection import permutation_importance
# r = permutation_importance(model, X_val, y_val,
#                            n_repeats=30,
#                            random_state=0)