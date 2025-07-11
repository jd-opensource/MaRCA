from audioop import mul
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def allocate_actions(state_dict, action_dict, value_list):
    n_si_remaining = state_dict.copy()
    q_aj_remaining = action_dict.copy()
    assignments = {}

    sorted_value_list = sorted(value_list, key=lambda x: x[2], reverse=True)

    for item in tqdm(sorted_value_list, desc='Processing'):
        si, aj, score = item
        n_si = n_si_remaining.get(si, 0)
        q_aj = q_aj_remaining.get(aj, 0)
        assign_num = min(n_si, q_aj)
        if assign_num > 0:
            assignments[(si, aj)] = assignments.get((si, aj), 0) + assign_num
            n_si_remaining[si] -= assign_num
            q_aj_remaining[aj] -= assign_num
        if si in n_si_remaining and n_si_remaining[si] == 0:
            del n_si_remaining[si]
        if aj in q_aj_remaining and q_aj_remaining[aj] == 0:
            del q_aj_remaining[aj]
        if not n_si_remaining or not q_aj_remaining:
            break

    new_data = []
    for (si, aj), count in assignments.items():
        new_data.append([si, aj, count])

    final_action_count = {}
    for _, aj, count in new_data:
        final_action_count[aj] = final_action_count.get(aj, 0) + count

    print("\n ori action_dict: ", action_dict)
    print("final_action_count: ", final_action_count)

    final_state_count = {}
    for si, _, count in new_data:
        final_state_count[si] = final_state_count.get(si, 0) + count

    print("\n ori state_dict: ", state_dict)
    print("final_state_count: ", final_state_count)

    return new_data, final_action_count, final_state_count


def get_predict(predict_dir, data_regex, predict_header):
    '''
    read predict result, return df
    '''
    print('predict result dir:', predict_dir)
    print('PWD:', os.getcwd())
    all_files = glob.glob(predict_dir + data_regex)
    print(all_files)
    head = predict_header.split(',')
    li = []
    for filename in all_files:
        print('load file:' + filename)
        try:
            df = pd.read_table(filename, index_col=None, names=head)
            li.append(df)
        except pd.errors.EmptyDataError as e:
            continue
    muli_df = pd.concat(li, axis=0, ignore_index=True)
    return muli_df


def process_predict_data(df, predictor_col='a5_output'):
    df['state'] = df['uv'].astype(str) + '_' + df['w'].astype(str) + '_' + df['g'].astype(str) + '_' + df['i'].astype(
        str)
    df['action'] = df['a1'].astype(str) + '_' + df['a2'].astype(str) + '_' + df['a3'].astype(str) + '_' + df[
        'a4'].astype(str) + '_' + df['a5'].astype(str)

    value_list = df[['state', 'action', predictor_col]].values.tolist()
    return value_list


def get_action_value(path, predictor_col, state_dict, action_dict):
    df_ori = pd.read_csv(path,
                         header='t,uv,w,g,i,a1,a2,a3,a4,a5,a1_output,a2_output,a3_output,a4_output,a5_output,q_re_output,q_pre_output,q_final_output,consumption')
    value_list = process_predict_data(df_ori, predictor_col)
    new_data, final_action_count, final_state_count = allocate_actions(state_dict, action_dict, value_list)
    final_data = [['%s_%s' % (i[0], i[1]), i[2]] for i in new_data]
    df = pd.DataFrame(final_data)
    return df


def get_expert_score(path, predictor_col):
    df_ori = pd.read_csv(path,
                         header='t,uv,w,g,i,a1,a2,a3,a4,a5,a1_output,a2_output,a3_output,a4_output,a5_output,q_re_output,q_pre_output,q_final_output,consumption')
    value_list = process_predict_data(df_ori, predictor_col)
    new_expert_list = [['%s_%s' % (i[0], i[1]), i[2]] for i in value_list]
    new_expert_dict = {item[0]: item[1] for item in new_expert_list}
    df = pd.DataFrame(list(new_expert_dict.items()), columns=['key', 'value'])
    return df


def score_with_expertmodel(new_expert_dict, expid, allocate_data):
    allocate_data = allocate_data.reset_index(drop=True)
    allocate_data = allocate_data.values.tolist()
    allocate_dict = {item[0]: item[1] for item in allocate_data}
    total_sum = sum(new_expert_dict.get(key, 0) * allocate_dict[key] for key in allocate_dict)
    print("%s total scores is: %s" % (expid, total_sum))


def main():
    action_df = pd.read_csv("action_value_data_generate_action_sample.csv", header=None,
                            names=['a1', 'a2', 'a3', 'a4', 'a5', 'an', 'num'])
    action_df['key'] = action_df[['a1', 'a2', 'a3', 'a4', 'a5']].apply(lambda row: '_'.join(row.astype(str)), axis=1)
    action_dict = pd.Series(action_df['an'].values, index=action_df['key']).to_dict()
    state_df = pd.read_csv("action_value_data_generate_state_sample.csv", header=None,
                           names=['uv', 'w', 'g', 'i', 'sn', 'num'])
    state_df['key'] = state_df['uv'].astype(str) + '_' + state_df['w'].astype(str) + '_' + state_df['g'].astype(
        str) + '_' + state_df['i'].astype(str)
    state_dict = pd.Series(state_df['sn'].values, index=state_df['key']).to_dict()

    predictor_col = 'a5_output'

    # read expert data from processed csv, or process with predict data from expert model
    expert_df = pd.read_csv("expert-data.csv")
    # expert_df = get_expert_score('expert_path',predictor_col)

    expert_df['value'].astype(float)
    new_expert_dict = expert_df.set_index('key')['value'].to_dict()

    ####### allocate experiment state&action with greedy score-based prioritization #######

    # allocate states with predict data
    # dqn_df = get_action_value('dqn_path',predictor_col, state_dict, action_dict)
    # or pre-allocated result from processed csv
    dqn_df = pd.read_csv("dqn-pred-data.csv", index_col=0)

    dqn_df = dqn_df.reset_index(drop=True)
    score_with_expertmodel(new_expert_dict, 'dqn', dqn_df)

    # same for marca, process or read pre-allocated result from processed csv
    # marca_df = get_action_value('marca_path',predictor_col, state_dict, action_dict)
    marca_df = pd.read_csv("marca-pred-data.csv", index_col=0)
    marca_df = marca_df.reset_index(drop=True)
    score_with_expertmodel(new_expert_dict, 'marca', marca_df)

if __name__ == '__main__':
    main()