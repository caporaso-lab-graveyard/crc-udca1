import scipy.stats
import pandas as pd

from statsmodels.sandbox.stats.multicomp import multipletests

def series_corr(s1, s2, metric=scipy.stats.spearmanr):
    df = pd.concat([s1, s2], axis=1).dropna()
    return metric(df.iloc[:, 0], df.iloc[:, 1])

def pairwise_corr_intersect(df1, df2, metric=scipy.stats.spearmanr, multipletests_method='fdr_bh'):
    data = []
    index = []
    for e in set(df1.columns) & set(df2.columns):
        index.append(e)
        data.append(series_corr(df1[e], df2[e]))
    result = pd.DataFrame(data, index=index, columns=['r', 'p'])
    result['q'] = multipletests(result['p'], method=multipletests_method)[1]
    return result

def pairwise_corr_all(df1, df2=None, metric=scipy.stats.spearmanr, multipletests_method='fdr_bh'):
    if df2 is None:
        df2 = df1
    data = []
    index = []
    for i in df1.columns:
        for j in df2.columns:
            index.append((i,j))
            data.append(series_corr(df1[i], df2[j]))
    result = pd.DataFrame(data, index=index, columns=['r', 'p'])
    result['q'] = multipletests(result['p'], method=multipletests_method)[1]
    return result

def get_group_pairs(df, group_value, individual_id_category='ptid',
                    group_category='treatmentgroup', state_category='visit',
                    state_values=['pre', 'post'], verbose=True):
    results = []
    group_members = df[group_category] == group_value
    group_md = df[group_members]
    for individual_id in set(group_md[individual_id_category]):
        result = []
        for state_value in state_values:
            individual_at_state_idx = group_md[
                (df[state_category] == state_value) & (df[individual_id_category] == individual_id)].index
            if len(individual_at_state_idx) > 1:
                if verbose:
                    print("Multiple values for %s %s at %s %s (%s)" %
                          (individual_id_category, individual_id, state_category, state_value, ' '.join(individual_at_state_idx)))
                break
            elif len(individual_at_state_idx) == 0:
                if verbose:
                    print("No values for %s %s at %s %s" %
                          (individual_id_category, individual_id, state_category, state_value))
                break
            else:
                result.append(individual_at_state_idx[0])
        if len(result) == len(state_values):
            results.append(tuple(result))
    return results
