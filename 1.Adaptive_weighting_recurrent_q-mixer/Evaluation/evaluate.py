from audioop import mul
import os
import glob
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import roc_auc_score

def printdf(df):
    print(df.columns)
    print('col number:'+ str(len(df.columns)))
    print('df data number:', len(df))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.head(5))

def get_predict(predict_dir,predict_header):
    print('predict result dir:', predict_dir)
    print('PWD:',os.getcwd())
    all_files = glob.glob(predict_dir + 'result_*.result')
    print(all_files)
    head = predict_header.split(',')
    head = head + ['a1_output', 'a2_output', 'a3_output', 'a4_output', 'a5_output','q_re_output','q_pre_output','q_final_output', 'consumption']
    li = []
    for filename in all_files:
        print('load file:' + filename)
        df = pd.read_table(filename, index_col=None, names=head, error_bad_lines=False) #.drop(columns=['kRequestDate','kAdPlanId'])
        df['consumption'] = df['consumption'] / 6
        li.append(df)
    muli_df = pd.concat(li, axis=0, ignore_index=True)
    printdf(muli_df)
    return muli_df

def spearman_corr(df, predict_col='offline', label_col="label"):
    return df[[predict_col, label_col]].corr(method="spearman").iloc[0, 1]

def getslice(df, bucketKey): 
    offlineSpearman_a1 = df.groupby(bucketKey).apply(spearman_corr, predict_col='a1_output', label_col="consumption").to_frame('offline_spearman_a1')
    offlineSpearman_a2 = df.groupby(bucketKey).apply(spearman_corr, predict_col='a2_output', label_col="consumption").to_frame('offline_spearman_a2')
    offlineSpearman_a3 = df.groupby(bucketKey).apply(spearman_corr, predict_col='a3_output', label_col="consumption").to_frame('offline_spearman_a3')
    offlineSpearman_a4 = df.groupby(bucketKey).apply(spearman_corr, predict_col='a4_output', label_col="consumption").to_frame('offline_spearman_a4')
    offlineSpearman_a5 = df.groupby(bucketKey).apply(spearman_corr, predict_col='a5_output', label_col="consumption").to_frame('offline_spearman_a5')
    offlineSpearman_q_re= df.groupby(bucketKey).apply(spearman_corr, predict_col='q_re_output', label_col="consumption").to_frame('offline_spearman_q_re')
    offlineSpearman_q_pre = df.groupby(bucketKey).apply(spearman_corr, predict_col='q_pre_output', label_col="consumption").to_frame('offline_spearman_q_pre')
    offlineSpearman_q_final = df.groupby(bucketKey).apply(spearman_corr, predict_col='q_final_output', label_col="consumption").to_frame('offline_spearman_q_final')
    
    groupres = df.groupby(bucketKey,as_index=False).agg({'a1_output':'sum', 'a2_output':'sum', 'a3_output':'sum', 'a4_output':'sum', 'a5_output':'sum','q_re_output':'sum','q_pre_output':'sum','q_final_output':'sum', 'consumption':'sum'})
    res = groupres.set_index(bucketKey).join(offlineSpearman_a1).join(offlineSpearman_a2).join(offlineSpearman_a3).join(offlineSpearman_a4).join(offlineSpearman_a5).join(offlineSpearman_q_re).join(offlineSpearman_q_pre).join(offlineSpearman_q_final)
    return res
    
def calculateAbsPE(df, ctr='CTR', pctr='offline_pCTR',abspecol='offline_abs_PE'):
    df[abspecol] = df[[ctr,pctr]].max(axis=1) / (df[[ctr,pctr]].min(axis=1)) - 1
    return df
def calculatePE(df, ctr='CTR', pctr='offline_pCTR',pecol='offline_PE'):
    df[pecol] = df[pctr]/ (df[ctr]) - 1
    return df
    
def getmetric(df):
    
    df = calculatePE(df,'consumption','a1_output','offline_PE_a1')
    df = calculateAbsPE(df,'consumption','a1_output','offline_abs_PE_a1')

    df = calculatePE(df,'consumption','a2_output','offline_PE_a2')
    df = calculateAbsPE(df,'consumption','a2_output','offline_abs_PE_a2')
    
    df = calculatePE(df,'consumption','a3_output','offline_PE_a3')
    df = calculateAbsPE(df,'consumption','a3_output','offline_abs_PE_a3')
    
    df = calculatePE(df,'consumption','a4_output','offline_PE_a4')
    df = calculateAbsPE(df,'consumption','a4_output','offline_abs_PE_a4')
    
    df = calculatePE(df,'consumption','a5_output','offline_PE_a5')
    df = calculateAbsPE(df,'consumption','a5_output','offline_abs_PE_a5')

    df = calculatePE(df,'consumption','q_re_output','offline_PE_q_re')
    df = calculateAbsPE(df,'consumption','q_re_output','offline_abs_PE_q_re')

    df = calculatePE(df,'consumption','q_pre_output','offline_PE_q_pre')
    df = calculateAbsPE(df,'consumption','q_pre_output','offline_abs_PE_q_pre')

    df = calculatePE(df,'consumption','q_final_output','offline_PE_q_final')
    df = calculateAbsPE(df,'consumption','q_final_output','offline_abs_PE_q_final')
    
    return df

def filternoise(df, consumptionTH=-1):
    return df[(df['consumption']>=consumptionTH)], df[~(df['consumption']>=consumptionTH)]

def AggBinId(df, bucketKey='t', weightcol='consumption'):
    def add_weighted_columns(suffix):
        df[f'weight_offline_PE_{suffix}'] = df[f'offline_PE_{suffix}'] * df[weightcol]
        df[f'weight_offline_abs_PE_{suffix}'] = df[f'offline_abs_PE_{suffix}'] * df[weightcol]
        df[f'weight_offline_spearman_{suffix}'] = df[f'offline_spearman_{suffix}'] * df[weightcol]
    for suffix in ['a1', 'a2', 'a3', 'a4', 'a5']:
        add_weighted_columns(suffix)
    for suffix in ['q_re', 'q_pre', 'q_final']:
        df[f'weight_offline_PE_{suffix}'] = df[f'offline_PE_{suffix}'] * df[weightcol]
        df[f'weight_offline_abs_PE_{suffix}'] = df[f'offline_abs_PE_{suffix}'] * df[weightcol]
        df[f'weight_offline_spearman_{suffix}'] = df[f'offline_spearman_{suffix}'] * df[weightcol]

    df['weightcol'] = df[weightcol]
    
    aggdf = df.groupby(bucketKey).agg(
        {
            'weightcol': [('weightsum', 'sum')],
            'consumption': [('consumption', 'sum')],

            'a1_output': [('a1_output', 'sum')],
            'weight_offline_spearman_a1': [('weight_offline_spearman_a1', 'sum')],
            'weight_offline_PE_a1': [('weight_sum_offline_PE_a1', 'sum')],
            'weight_offline_abs_PE_a1': [('weight_sum_offline_abs_PE_a1', 'sum')],

            'a2_output': [('a2_output', 'sum')],
            'weight_offline_spearman_a2': [('weight_offline_spearman_a2', 'sum')],
            'weight_offline_PE_a2': [('weight_sum_offline_PE_a2', 'sum')],
            'weight_offline_abs_PE_a2': [('weight_sum_offline_abs_PE_a2', 'sum')],

            'a3_output': [('a3_output', 'sum')],
            'weight_offline_spearman_a3': [('weight_offline_spearman_a3', 'sum')],
            'weight_offline_PE_a3': [('weight_sum_offline_PE_a3', 'sum')],
            'weight_offline_abs_PE_a3': [('weight_sum_offline_abs_PE_a3', 'sum')],

            'a4_output': [('a4_output', 'sum')],
            'weight_offline_spearman_a4': [('weight_offline_spearman_a4', 'sum')],
            'weight_offline_PE_a4': [('weight_sum_offline_PE_a4', 'sum')],
            'weight_offline_abs_PE_a4': [('weight_sum_offline_abs_PE_a4', 'sum')],

            'a5_output': [('a5_output', 'sum')],
            'weight_offline_spearman_a5': [('weight_offline_spearman_a5', 'sum')],
            'weight_offline_PE_a5': [('weight_sum_offline_PE_a5', 'sum')],
            'weight_offline_abs_PE_a5': [('weight_sum_offline_abs_PE_a5', 'sum')],

            'q_re_output': [('q_re_output', 'sum')],
            'weight_offline_spearman_q_re': [('weight_offline_spearman_q_re', 'sum')],
            'weight_offline_PE_q_re': [('weight_sum_offline_PE_q_re', 'sum')],
            'weight_offline_abs_PE_q_re': [('weight_sum_offline_abs_PE_q_re', 'sum')],

            'q_pre_output': [('q_pre_output', 'sum')],
            'weight_offline_spearman_q_pre': [('weight_offline_spearman_q_pre', 'sum')],
            'weight_offline_PE_q_pre': [('weight_sum_offline_PE_q_pre', 'sum')],
            'weight_offline_abs_PE_q_pre': [('weight_sum_offline_abs_PE_q_pre', 'sum')],

            'q_final_output': [('q_final_output', 'sum')],
            'weight_offline_spearman_q_final': [('weight_offline_spearman_q_final', 'sum')],
            'weight_offline_PE_q_final': [('weight_sum_offline_PE_q_final', 'sum')],
            'weight_offline_abs_PE_q_final': [('weight_sum_offline_abs_PE_q_final', 'sum')]
        }
    )
    
    aggdf.columns = aggdf.columns.get_level_values(1)

    for suffix in ['a1', 'a2', 'a3', 'a4', 'a5']:
        aggdf[f'weight_offline_spearman_{suffix}'] = aggdf[f'weight_offline_spearman_{suffix}'] / aggdf['weightsum']
        aggdf[f'weight_offline_PE_{suffix}'] = aggdf[f'weight_sum_offline_PE_{suffix}'] / aggdf['weightsum']
        aggdf[f'weight_offline_abs_PE_{suffix}'] = aggdf[f'weight_sum_offline_abs_PE_{suffix}'] / aggdf['weightsum']

    for suffix in ['q_re', 'q_pre', 'q_final']:
        aggdf[f'weight_offline_spearman_{suffix}'] = aggdf[f'weight_offline_spearman_{suffix}'] / aggdf['weightsum']
        aggdf[f'weight_offline_PE_{suffix}'] = aggdf[f'weight_sum_offline_PE_{suffix}'] / aggdf['weightsum']
        aggdf[f'weight_offline_abs_PE_{suffix}'] = aggdf[f'weight_sum_offline_abs_PE_{suffix}'] / aggdf['weightsum']
    
    return aggdf


def eval_badcase(result_dir, threshold):
    t_w_metric = pd.read_csv(result_dir + "metric_t|uv_.tsv",sep='\t')
    t_w_metric_bad = t_w_metric[t_w_metric['offline_abs_PE'] >= threshold]
    badcase_rate = t_w_metric_bad['impression'].sum() / t_w_metric['impression'].sum() 
    return badcase_rate
    

def getslice_with_bucket_spearman(df, bucketKey, weightcol='consumption_mean'):
    group_means = df.groupby(bucketKey).agg(
        a1_output_mean=('a1_output', 'mean'),
        a2_output_mean=('a2_output', 'mean'),
        a3_output_mean=('a3_output', 'mean'),
        a4_output_mean=('a4_output', 'mean'),
        a5_output_mean=('a5_output', 'mean'),
        q_re_output_mean=('q_re_output', 'mean'),
        q_pre_output_mean=('q_pre_output', 'mean'),
        q_final_output_mean=('q_final_output', 'mean'),
        consumption_mean=('consumption', 'mean')
    ).reset_index()

    output_columns = ['a1', 'a2', 'a3', 'a4', 'a5']
    spearman_results = {}

    for col in output_columns:
        spearman_results[f'{col}_bucket_spearman'] = group_means.groupby('t').apply(
            spearman_corr, predict_col=f'{col}_output_mean', label_col="consumption_mean"
        ).to_frame(f'{col}_bucket_spearman')

    # Calculating spearman for q_re, q_pre, q_final
    for col in ['q_re', 'q_pre', 'q_final']:
        spearman_results[f'{col}_bucket_spearman'] = group_means.groupby('t').apply(
            spearman_corr, predict_col=f'{col}_output_mean', label_col="consumption_mean"
        ).to_frame(f'{col}_bucket_spearman')

    for col in output_columns:
        group_means = calculatePE(group_means, 'consumption_mean', f'{col}_output_mean', f'{col}_bucket_PE')
        group_means = calculateAbsPE(group_means, 'consumption_mean', f'{col}_output_mean', f'{col}_bucket_abs_PE')

    for col in ['q_re', 'q_pre', 'q_final']:
        group_means = calculatePE(group_means, 'consumption_mean', f'{col}_output_mean', f'{col}_bucket_PE')
        group_means = calculateAbsPE(group_means, 'consumption_mean', f'{col}_output_mean', f'{col}_bucket_abs_PE')

    for col in output_columns + ['q_re', 'q_pre', 'q_final']:
        group_means[f'{col}_weight_bucket_PE'] = group_means[f'{col}_bucket_PE'] * group_means[weightcol]
        group_means[f'{col}_weight_bucket_abs_PE'] = group_means[f'{col}_bucket_abs_PE'] * group_means[weightcol]

    group_means['weightcol'] = group_means[weightcol]

    agg_group_means = group_means.groupby('t').agg(
        {
            'weightcol': 'sum',  # Ensure the sum is calculated for weightsum
        }
    )
    agg_group_means.rename(columns={'weightcol': 'weightsum'}, inplace=True)

    for col in output_columns + ['q_re', 'q_pre', 'q_final']:
        agg_group_means[f'{col}_weight_sum_bucket_PE'] = group_means.groupby('t')[f'{col}_weight_bucket_PE'].sum()
        agg_group_means[f'{col}_weight_sum_bucket_abs_PE'] = group_means.groupby('t')[f'{col}_weight_bucket_abs_PE'].sum()

    for col in output_columns + ['q_re', 'q_pre', 'q_final']:
        agg_group_means[f'{col}_weight_bucket_PE'] = agg_group_means[f'{col}_weight_sum_bucket_PE'] / agg_group_means['weightsum']
        agg_group_means[f'{col}_weight_bucket_abs_PE'] = agg_group_means[f'{col}_weight_sum_bucket_abs_PE'] / agg_group_means['weightsum']

    for col in spearman_results:
        agg_group_means = agg_group_means.join(spearman_results[col])

    agg_group_means = agg_group_means.reset_index()

    selected_columns = ['t']
    for col in output_columns + ['q_re', 'q_pre', 'q_final']:
        selected_columns += [
            f'{col}_bucket_spearman', f'{col}_weight_bucket_PE', f'{col}_weight_bucket_abs_PE'
        ]

    result = agg_group_means[selected_columns].sort_values('t').copy()
    result['bucketKey'] = [bucketKey] * len(result)
    return result

def do_eval(input_dir, result_dir, predict_header, eval_slice):
    df = get_predict(input_dir, predict_header)
    t_metric = None
    t_uv_metric = None
    t_w_g_metric = None
    t_a1_metric = None
    t_a2_metric = None
    t_a3_metric = None
    t_a4_metric = None
    t_a5_metric = None
    for i in eval_slice.split(','):
        if i == 't':
            print('Process Slice:\t',i)
            metric1 = getslice(df, bucketKey=i.split('|')).sort_values('consumption',ascending=False)#+[offline_spearman]
            metric2, metric2_filtered = filternoise(getmetric(metric1).reset_index().sort_values('t')) #+[offline_PE,offline_abs_PE]
            printdf(metric1)
            printdf(metric2)
            printdf(metric2_filtered)
            t_metric = metric2
            t_metric.set_index('t',inplace = True)	
        if i == 't|uv':
            t_uv_metric = metric2
        if i == 't|w|g':
            t_w_g_metric = metric2
        if i == 't|a1':
            t_a1_metric = metric2
        if i == 't|a2':
            t_a2_metric = metric2
        if i == 't|a3':
            t_a3_metric = metric2
        if i == 't|a4':
            t_a4_metric = metric2
        if i == 't|a5':
            t_a5_metric = metric2
        print('Write Metric Result:\t',i)
        metric_path = result_dir + 'metric_' + i + '_.tsv'
        filtered_metric_path = result_dir + '_filteredmetric_' + i + '_.tsv'
        metric2.to_csv(metric_path,sep='\t')
        metric2_filtered.to_csv(filtered_metric_path,sep='\t')

    metrics_dict = {
    't_uv_metric': t_uv_metric,
    't_w_g_metric': t_w_g_metric,
    't_a1_metric' : t_a1_metric,
    't_a2_metric' : t_a2_metric,
    't_a3_metric' : t_a3_metric,
    't_a4_metric' : t_a4_metric,
    't_a5_metric' : t_a5_metric
    }
    li = []
    for i in eval_slice.split(','):
        if i=='t':
            continue
        bucket_name = i.replace('|', '_')
        metric_name = metrics_dict.get(f"{bucket_name}_metric")
        
        t_agg_metric = AggBinId(df=metric_name, bucketKey='t', weightcol='consumption')
        
        bucket_spearman = getslice_with_bucket_spearman(df=df, bucketKey=i.split('|'))
        print(bucket_spearman)
        li.append(bucket_spearman)
        
        result = t_agg_metric[['consumption',
        'a1_output','weight_offline_spearman_a1', 'weight_offline_PE_a1','weight_offline_abs_PE_a1',
                               'a2_output','weight_offline_spearman_a2', 'weight_offline_PE_a2','weight_offline_abs_PE_a2',
                               'a3_output','weight_offline_spearman_a3', 'weight_offline_PE_a3','weight_offline_abs_PE_a3',
                               'a4_output','weight_offline_spearman_a4', 'weight_offline_PE_a4','weight_offline_abs_PE_a4',
                               'a5_output','weight_offline_spearman_a5', 'weight_offline_PE_a5','weight_offline_abs_PE_a5',
                               'q_re_output','weight_offline_spearman_q_re', 'weight_offline_PE_q_re','weight_offline_abs_PE_q_re',
                               'q_pre_output','weight_offline_spearman_q_pre', 'weight_offline_PE_q_pre','weight_offline_abs_PE_q_pre',
                               'q_final_output','weight_offline_spearman_q_final', 'weight_offline_PE_q_final','weight_offline_abs_PE_q_final',
                              ]].join(t_metric[['offline_spearman_a1','offline_spearman_a2','offline_spearman_a3','offline_spearman_a4','offline_spearman_a5','offline_spearman_q_re','offline_spearman_q_pre','offline_spearman_q_final']]).sort_values('t')
        
        
        printdf(t_agg_metric)
        printdf(result)
        print('Write Final Metric Result:\t')
        
        result_path = result_dir + 'finallResul' + bucket_name + '.tsv'
        result.to_csv(result_path,sep='\t')
        print("eval done, write file to:", result_path)
    print('--------spearman------------')
    df_bucket = pd.concat(li, axis=0, ignore_index=True)
    result_spearman_path = result_dir + 'bucket_spearman.tsv'
    df_bucket.to_csv(result_spearman_path, sep='\t')
    print("eval done, write file to:", result_spearman_path)


input_dir= './'
output_dir ='eval/'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

predict_header='t,uv,w,g,i,a1,a2,a3,a4,a5'
evalSlice = 't,t|uv,t|w|g,t|a1,t|a2,t|a3,t|a4,t|a5'
do_eval(input_dir, output_dir, predict_header, evalSlice)
