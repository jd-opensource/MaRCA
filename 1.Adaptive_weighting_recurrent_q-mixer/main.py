import numpy as np
import tensorflow as tf
import csv
from config import MODEL, EPOCHS, BATCH_SIZE, TARGET_UPDATE_FREQ, STATE_SPACE, HIDDEN_SIZE, RE_ACTION_SPACE, \
    PRE_ACTION_SPACE, SINGLE_AGENT_ACTION_DIM, LEARNING_RATE, ABLATION_FLAG, ABLATION_MODEL, OPTIMIZER
from model import (
    build_awrq, build_hypernet_awrq, train_step_awrq,
    build_seq_dqn, train_step_seq_dqn, train_step_hyper_awrq, train_step_double_dqn, build_seq_drqn,
    train_step_seq_drqn, build_seq_emdqn, train_step_seq_emdqn,
    build_seq_remdqn, train_step_seq_remdqn, build_drqn, build_hypernet, train_step_drqn, train_step_hyper,
    train_step_vdn, train_step_weighted,
    train_step_awrq_aw, train_step_awrq_vgca
)


def get_optimizer():
    if OPTIMIZER == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    else:
        raise ValueError(f"Unsupported Optimizer: {OPTIMIZER}")

def hash_state(state):
    return [hash(val) for val in state]

def load_data(file):
    S, A, R = [], [], []
    with open(file) as f:
        for row in csv.reader(f):
            vals = []
            for i, val in enumerate(row):
                try:
                    vals.append(float(val))
                except ValueError:
                    vals.append(val)

            state = vals[:STATE_SPACE]
            state = hash_state(state)
            S.append(state)

            A.append(list(map(int, vals[STATE_SPACE:STATE_SPACE + len(RE_ACTION_SPACE) + 1])))
            R.append(vals[-1])

    S = np.array(S, np.float32)
    A = np.array(A, np.int32)
    R = np.array(R, np.float32)

    mean, std = S.mean(0), S.std(0) + 1e-6
    return (S - mean) / std, A, R

states, actions, rewards = load_data('data.csv')
N = len(states)
batches = (N + BATCH_SIZE - 1) // BATCH_SIZE

if (MODEL == 'AWRQ' and ABLATION_FLAG == False):
    awrq_main = build_awrq(RE_ACTION_SPACE)
    awrq_main_target = build_awrq(RE_ACTION_SPACE)
    awrq_a5 = build_awrq(PRE_ACTION_SPACE)
    hyper = build_hypernet_awrq()

    opt_main = get_optimizer()
    opt_a5 = get_optimizer()
    opt_hyper = get_optimizer()


    def update_main_target():
        awrq_main_target.set_weights(awrq_main.get_weights())


    update_main_target()

    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_main = tot_a5 = tot_mix = grads = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq_main = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            lm, h_new, q4 = train_step_awrq(
                awrq_main, awrq_main_target, opt_main,
                tf.convert_to_tensor(seq_main),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), RE_ACTION_SPACE)
            tot_main += lm.numpy()
            # a5 DRQN (single step)
            seq_a5 = bs[:, None, :]
            la5, _, q5 = train_step_awrq(
                awrq_a5, None, opt_a5,
                tf.convert_to_tensor(seq_a5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), PRE_ACTION_SPACE)
            tot_a5 += la5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                update_main_target()
            lm, grad = train_step_hyper_awrq(
                hyper, opt_hyper,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                q4, q5,
                tf.convert_to_tensor(br))
            tot_mix += lm.numpy()
            grads += grad.numpy()
        print(
            f"Epoch {ep}: Main={tot_main / batches:.4f}, A5={tot_a5 / batches:.4f}, Mix={tot_mix / batches:.4f},  Grads={grads / batches:.4f}")
    print("Done")

elif (MODEL == 'DQN' and ABLATION_FLAG == False):
    seq_dqn = build_seq_dqn()
    seq_dqn_target = build_seq_dqn()
    seq_dqn_target.set_weights(seq_dqn.get_weights())
    dqn_opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_loss = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            loss = train_step_seq_dqn(
                seq_dqn, seq_dqn_target, dqn_opt,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                tf.convert_to_tensor(br)
            )
            tot_loss += loss.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                seq_dqn_target.set_weights(seq_dqn.get_weights())
        print(f"Epoch {ep}/{EPOCHS}: SeqDQN Loss={tot_loss / batches:.4f}")
elif (MODEL == 'DoubleDQN' and ABLATION_FLAG == False):
    seq_dqn = build_seq_dqn()
    seq_dqn_target = build_seq_dqn()
    seq_dqn_target.set_weights(seq_dqn.get_weights())
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N);
        tot = 0
        for i in range((N + BATCH_SIZE - 1) // BATCH_SIZE):
            b = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[b], actions[b], rewards[b]
            loss = train_step_double_dqn(
                seq_dqn, seq_dqn_target, opt,
                tf.convert_to_tensor(bs), tf.convert_to_tensor(ba), tf.convert_to_tensor(br)
            )
            tot += loss.numpy()
            if i % TARGET_UPDATE_FREQ == 0: seq_dqn_target.set_weights(seq_dqn.get_weights())
        print(f"Epoch{ep}:DoubleDQN Loss={tot:.4f}")

elif (MODEL == 'DRQN' and ABLATION_FLAG == False):
    seq_drqn = build_seq_drqn()
    seq_drqn_target = build_seq_drqn()
    seq_drqn_target.set_weights(seq_drqn.get_weights())
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_loss = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE) + 1, axis=1)
            loss = train_step_seq_drqn(
                seq_drqn, seq_drqn_target, opt,
                tf.convert_to_tensor(seq),
                tf.convert_to_tensor(ba),
                tf.convert_to_tensor(br)
            )
            tot_loss += loss.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                seq_drqn_target.set_weights(seq_drqn.get_weights())
        print(f"Epoch {ep}/{EPOCHS}: DRQN Loss={tot_loss / batches:.4f}")

elif (MODEL == 'EMDQN' and ABLATION_FLAG == False):
    seq_emdqn = build_seq_emdqn()
    seq_emdqn_target = build_seq_emdqn()
    seq_emdqn_target.set_weights(seq_emdqn.get_weights())
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_loss = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            loss = train_step_seq_emdqn(
                seq_emdqn, seq_emdqn_target, opt,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                tf.convert_to_tensor(br)
            )
            tot_loss += loss.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                seq_emdqn_target.set_weights(seq_emdqn.get_weights())
        print(f"Epoch {ep}/{EPOCHS}: EMDQN Loss={tot_loss / batches:.4f}")

elif (MODEL == 'REMDQN' and ABLATION_FLAG == False):
    seq_rem = build_seq_remdqn()
    seq_rem_t = build_seq_remdqn()
    seq_rem_t.set_weights(seq_rem.get_weights())
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_loss = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            loss = train_step_seq_remdqn(
                seq_rem, seq_rem_t, opt,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                tf.convert_to_tensor(br)
            )
            tot_loss += loss.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                seq_rem_t.set_weights(seq_rem.get_weights())
        print(f"Epoch {ep}/{EPOCHS}: REMDQN Loss={tot_loss / batches:.4f}")
elif (MODEL == 'QMIX' and ABLATION_FLAG == False):
    drqn_main = build_drqn(RE_ACTION_SPACE)
    drqn_main_t = build_drqn(RE_ACTION_SPACE)
    drqn_a5 = build_drqn(PRE_ACTION_SPACE)
    hyper = build_hypernet()
    opt_m = get_optimizer()
    opt_5 = get_optimizer()
    opt_h = get_optimizer()
    drqn_main_t.set_weights(drqn_main.get_weights())

    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_m = tot5 = tot_h = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq4 = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            lm, _ = train_step_drqn(
                drqn_main, drqn_main_t, opt_m,
                tf.convert_to_tensor(seq4),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                br,
                tf.zeros((len(idx), HIDDEN_SIZE)),
                RE_ACTION_SPACE
            )
            tot_m += lm.numpy()
            seq5 = bs[:, None, :]
            l5, _ = train_step_drqn(
                drqn_a5, drqn_a5, opt_5,
                tf.convert_to_tensor(seq5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                br,
                tf.zeros((len(idx), HIDDEN_SIZE)),
                PRE_ACTION_SPACE
            )
            tot5 += l5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                drqn_main_t.set_weights(drqn_main.get_weights())
            out_main = drqn_main([
                tf.convert_to_tensor(seq4), tf.zeros((len(idx), HIDDEN_SIZE))
            ])
            q_list_main, h_tmp = out_main[:-1], out_main[-1]
            q4_logits = q_list_main[3]  # [B, dim4]
            q4_sel = tf.gather(q4_logits, ba[:, 3], axis=1, batch_dims=1)  # [B]
            out_a5 = drqn_a5([
                tf.convert_to_tensor(seq5), tf.zeros((len(idx), HIDDEN_SIZE))
            ])
            q_list_a5, h_tmp2 = out_a5[:-1], out_a5[-1]
            q5_logits = q_list_a5[0]  # [B, dim5]
            q5_sel = tf.gather(q5_logits, ba[:, -1], axis=1,
                               batch_dims=1)
            lh = train_step_hyper(
                hyper, opt_h,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                tf.expand_dims(q4_sel, 1),
                tf.expand_dims(q5_sel, 1),
                br
            )
            tot_h += lh.numpy()
        print(
            f"Epoch {ep}/{EPOCHS}: DRQN4={tot_m / batches:.4f}, DRQN5={tot5 / batches:.4f}, Hyper={tot_h / batches:.4f}")

elif (MODEL == 'VDN' and ABLATION_FLAG == False):
    drqn_main = build_drqn(RE_ACTION_SPACE)
    drqn_a5 = build_drqn(PRE_ACTION_SPACE)
    opt_m = get_optimizer()
    opt_5 = get_optimizer()
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_v = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq4 = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            seq5 = bs[:, None, :]
            loss_v = train_step_vdn(
                drqn_main, drqn_a5,
                opt_m, opt_5,
                tf.convert_to_tensor(seq4),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                tf.convert_to_tensor(seq5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                br,
                tf.zeros((len(idx), HIDDEN_SIZE))
            )
            tot_v += loss_v.numpy()
        print(f"Epoch {ep}/{EPOCHS}: VDN Loss={tot_v / batches:.4f}")
elif (MODEL == 'WEIGHTED_QMIX' and ABLATION_FLAG == False):
    drqn_main = build_drqn(RE_ACTION_SPACE)
    drqn_main_t = build_drqn(RE_ACTION_SPACE)
    drqn_a5 = build_drqn(PRE_ACTION_SPACE)
    hyper = build_hypernet()
    opt_m = get_optimizer()
    opt_5 = get_optimizer()
    opt_h = get_optimizer()
    drqn_main_t.set_weights(drqn_main.get_weights())
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_m = tot5 = tot_h = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq4 = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            seq5 = bs[:, None, :]
            lm, _ = train_step_drqn(
                drqn_main, drqn_main_t, opt_m,
                tf.convert_to_tensor(seq4),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                br,
                tf.zeros((len(idx), HIDDEN_SIZE)),
                RE_ACTION_SPACE
            )
            tot_m += lm.numpy()
            l5, _ = train_step_drqn(
                drqn_a5, drqn_a5, opt_5,
                tf.convert_to_tensor(seq5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                br,
                tf.zeros((len(idx), HIDDEN_SIZE)),
                PRE_ACTION_SPACE
            )
            tot5 += l5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                drqn_main_t.set_weights(drqn_main.get_weights())
            out_main = drqn_main([
                tf.convert_to_tensor(seq4), tf.zeros((len(idx), HIDDEN_SIZE))
            ])
            q_list_main, _ = out_main[:-1], out_main[-1]
            q4_sel = tf.gather(q_list_main[3], ba[:, 3], axis=1, batch_dims=1)
            out_a5 = drqn_a5([
                tf.convert_to_tensor(seq5), tf.zeros((len(idx), HIDDEN_SIZE))
            ])
            q_list_a5, _ = out_a5[:-1], out_a5[-1]
            q5_sel = tf.gather(q_list_a5[0], ba[:, -1], axis=1, batch_dims=1)
            lh = train_step_weighted(
                hyper, opt_h,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                tf.expand_dims(q4_sel, 1),
                tf.expand_dims(q5_sel, 1),
                br
            )
            tot_h += lh.numpy()
        print(
            f"Epoch {ep}/{EPOCHS}: DRQN4={tot_m / batches:.4f}, DRQN5={tot5 / batches:.4f}, Weighted_Hyper={tot_h / batches:.4f}")
elif (ABLATION_MODEL == 'AWRQ_w/o_AW' and ABLATION_FLAG == True):
    drqn_main = build_awrq(RE_ACTION_SPACE)
    drqn_main_target = build_awrq(RE_ACTION_SPACE)
    drqn_a5 = build_awrq(PRE_ACTION_SPACE)
    hyper = build_hypernet_awrq()

    opt_main = get_optimizer()
    opt_a5 = get_optimizer()
    opt_hyper = get_optimizer()
    def update_main_target():
        drqn_main_target.set_weights(drqn_main.get_weights())


    update_main_target()

    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_main = tot_a5 = tot_mix = grads = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq_main = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            lm, h_new, q4 = train_step_awrq_aw(
                drqn_main, drqn_main_target, opt_main,
                tf.convert_to_tensor(seq_main),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), RE_ACTION_SPACE)
            tot_main += lm.numpy()
            seq_a5 = bs[:, None, :]
            la5, _, q5 = train_step_awrq_aw(
                drqn_a5, None, opt_a5,
                tf.convert_to_tensor(seq_a5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), PRE_ACTION_SPACE)
            tot_a5 += la5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                update_main_target()
            lm, grad = train_step_hyper_awrq(
                hyper, opt_hyper,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                q4, q5,
                tf.convert_to_tensor(br))
            tot_mix += lm.numpy()
            grads += grad.numpy()
        print(
            f"Epoch {ep}: Main={tot_main / batches:.4f}, A5={tot_a5 / batches:.4f}, Mix={tot_mix / batches:.4f}, Grads={grads / batches:.4f}")
    print("Done")

elif (ABLATION_MODEL == 'AWRQ_w/o_VGCA' and ABLATION_FLAG == True):
    drqn_main = build_awrq(RE_ACTION_SPACE)
    drqn_main_target = build_awrq(RE_ACTION_SPACE)
    drqn_a5 = build_awrq(PRE_ACTION_SPACE)
    hyper = build_hypernet_awrq()

    opt_main = get_optimizer()
    opt_a5 = get_optimizer()
    opt_hyper = get_optimizer()


    def update_main_target():
        drqn_main_target.set_weights(drqn_main.get_weights())


    update_main_target()

    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_main = tot_a5 = tot_mix = grads = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq_main = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            lm, h_new, q4 = train_step_awrq_vgca(
                drqn_main, drqn_main_target, opt_main,
                tf.convert_to_tensor(seq_main),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), RE_ACTION_SPACE)
            tot_main += lm.numpy()
            seq_a5 = bs[:, None, :]
            la5, _, q5 = train_step_awrq_vgca(
                drqn_a5, None, opt_a5,
                tf.convert_to_tensor(seq_a5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), PRE_ACTION_SPACE)
            tot_a5 += la5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                update_main_target()
            lm, grad = train_step_hyper_awrq(
                hyper, opt_hyper,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                q4, q5,
                tf.convert_to_tensor(br))
            tot_mix += lm.numpy()
            grads += grad.numpy()
        print(
            f"Epoch {ep}: Main={tot_main / batches:.4f}, A5={tot_a5 / batches:.4f}, Mix={tot_mix / batches:.4f}, Grads={grads / batches:.4f}")
    print("Done")

elif (ABLATION_MODEL == 'AWRQ_w/o_SMC' and ABLATION_FLAG == True):
    drqn_main = build_awrq(RE_ACTION_SPACE)
    drqn_main_target = build_awrq(RE_ACTION_SPACE)
    drqn_a5 = build_awrq(PRE_ACTION_SPACE)
    hyper = build_hypernet()

    opt_main = get_optimizer()
    opt_a5 = get_optimizer()
    opt_hyper = get_optimizer()


    def update_main_target():
        drqn_main_target.set_weights(drqn_main.get_weights())


    update_main_target()

    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(1, EPOCHS + 1):
        perm = np.random.permutation(N)
        tot_main = tot_a5 = tot_mix = grads = 0.0
        for i in range(batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            bs, ba, br = states[idx], actions[idx], rewards[idx]
            seq_main = np.repeat(bs[:, None, :], len(RE_ACTION_SPACE), axis=1)
            lm, h_new, q4 = train_step_awrq(
                drqn_main, drqn_main_target, opt_main,
                tf.convert_to_tensor(seq_main),
                tf.convert_to_tensor(ba[:, :len(RE_ACTION_SPACE)]),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), RE_ACTION_SPACE)
            tot_main += lm.numpy()
            seq_a5 = bs[:, None, :]
            la5, _, q5 = train_step_awrq(
                drqn_a5, None, opt_a5,
                tf.convert_to_tensor(seq_a5),
                tf.convert_to_tensor(ba[:, -1:], tf.int32),
                tf.convert_to_tensor(br),
                tf.zeros((len(idx), HIDDEN_SIZE)), PRE_ACTION_SPACE)
            tot_a5 += la5.numpy()
            if i % TARGET_UPDATE_FREQ == 0:
                update_main_target()
            lm, grad = train_step_hyper_awrq(
                hyper, opt_hyper,
                tf.convert_to_tensor(bs),
                tf.convert_to_tensor(ba),
                q4, q5,
                tf.convert_to_tensor(br))
            tot_mix += lm.numpy()
            grads += grad.numpy()
        print(
            f"Epoch {ep}: Main={tot_main / batches:.4f}, A5={tot_a5 / batches:.4f}, Mix={tot_mix / batches:.4f}, Grads={grads / batches:.4f}")
    print("Done")

else:
    raise ValueError(f"Unsupported MODEL: {MODEL}")