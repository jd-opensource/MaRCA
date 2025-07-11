import tensorflow as tf
from tensorflow.keras import layers, models
from config import (GAMMA, HIDDEN_SIZE, STATE_SPACE,RE_ACTION_SPACE, PRE_ACTION_SPACE, ENSEMBLE_SIZE, EMBED_DIM,SINGLE_AGENT_ACTION_DIM, LEARNING_RATE,CLIP_GRADIENT,GRADIENT_CLIP_NORM,ACTIVATION_FUNCTION,
                    LOSS_FUNCTION,USE_BATCH_NORM,DROPOUT_RATE,WEIGHT_INITIALIZER)

def clip_gradients(optimizer, grads):
    if CLIP_GRADIENT:
        grads = [tf.clip_by_norm(g, GRADIENT_CLIP_NORM) for g in grads]
    return grads

def compute_loss(q_pred, target):
    if LOSS_FUNCTION == 'mse':
        return tf.reduce_mean((q_pred - target)**2)
    elif LOSS_FUNCTION == 'mae':
        return tf.reduce_mean(tf.abs(q_pred - target))
    elif LOSS_FUNCTION == 'huber':
        return tf.reduce_mean(tf.keras.losses.Huber()(q_pred, target))
    else:
        raise ValueError(f"Unsupported Loss Function: {LOSS_FUNCTION}")

def train_step_weighted(hyper, optimizer, states, actions, q4, q5, rewards):
    rewards = tf.cast(rewards, tf.float32)
    ohs = [tf.one_hot(actions[:,i], dim) for i, dim in enumerate(RE_ACTION_SPACE)]
    oh5 = tf.one_hot(actions[:,-1], PRE_ACTION_SPACE[0])
    inp = tf.concat([states] + ohs + [oh5], axis=1)
    with tf.GradientTape() as tape:
        w1_flat, b1, w2_flat, b2 = hyper(inp)
        B = tf.shape(states)[0]
        w1 = tf.reshape(w1_flat, [B, 2, EMBED_DIM])
        b1 = tf.reshape(b1, [B, EMBED_DIM])
        w2 = tf.reshape(w2_flat, [B, EMBED_DIM, 1])
        b2 = tf.reshape(b2, [B, 1])
        q_cat = tf.concat([q4, q5], axis=1)[:,None,:]
        hidden = tf.nn.relu(tf.matmul(q_cat, w1) + b1[:,None,:])
        q_tot = tf.matmul(hidden, w2) + b2[:,None]
        q_tot = tf.squeeze(q_tot, [1,2])
        per = compute_loss(q_tot,rewards)
        mask = tf.cast(q_tot > rewards, tf.float32)
        per_weighted = per * (1.0 - 0.5 * mask)
        loss = tf.reduce_mean(per_weighted)
    grads = tape.gradient(loss, hyper.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, hyper.trainable_variables))
    return loss

def train_step_vdn(drqn_main, drqn_a5, optimizer_main, optimizer_a5,
                   state_seq4, actions4, state_seq5, actions5, rewards, h0):
    rewards = tf.cast(rewards, tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        *q4_list, h4 = drqn_main([state_seq4, h0])
        q4_logits = q4_list[-1]
        q4_sel = tf.gather(q4_logits, actions4[:, -1], axis=1, batch_dims=1)
        *q5_list, h5 = drqn_a5([state_seq5, h0])
        q5_logits = q5_list[0]
        q5_sel = tf.gather(q5_logits, actions5[:, 0], axis=1, batch_dims=1)
        q_tot = q4_sel + q5_sel
        loss = compute_loss(q_tot,rewards)
    grads4 = tape.gradient(loss, drqn_main.trainable_variables)
    grads5 = tape.gradient(loss, drqn_a5.trainable_variables)
    grads_and_vars4 = [(g, v) for g, v in zip(grads4, drqn_main.trainable_variables) if g is not None]
    grads_and_vars5 = [(g, v) for g, v in zip(grads5, drqn_a5.trainable_variables) if g is not None]
    optimizer_main.apply_gradients(grads_and_vars4)
    optimizer_a5.apply_gradients(grads_and_vars5)
    del tape
    return loss

def build_drqn(step_dims):
    state_seq = layers.Input((None, STATE_SPACE), name='state_seq')
    h0 = layers.Input((HIDDEN_SIZE,), name='h0')
    rnn = layers.RNN(layers.GRUCell(HIDDEN_SIZE), return_sequences=True, return_state=True)
    seq_out, h_new = rnn(state_seq, initial_state=[h0])
    qs = []
    for t, dim in enumerate(step_dims):
        x = layers.Lambda(lambda x, i=t: x[:, i, :])(seq_out)
        if USE_BATCH_NORM:
            x = tf.keras.layers.BatchNormalization()(x)
        if DROPOUT_RATE > 0:
            x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        qs.append(layers.Dense(dim)(x))
    return models.Model([state_seq, h0], [*qs, h_new], name=f'DRQN_{len(step_dims)}')

def build_hypernet():
    tot_act = sum(RE_ACTION_SPACE) + sum(PRE_ACTION_SPACE)
    inp = layers.Input((STATE_SPACE + tot_act,), name='hyper_in')
    h = layers.Dense(128, activation=ACTIVATION_FUNCTION,kernel_initializer=WEIGHT_INITIALIZER)(inp)
    w1 = layers.Dense(EMBED_DIM*2, activation=lambda x: tf.abs(x))(h)
    b1 = layers.Dense(EMBED_DIM)(h)
    w2 = layers.Dense(EMBED_DIM, activation=lambda x: tf.abs(x))(h)
    b2 = layers.Dense(1)(h)
    return models.Model(inp, [w1, b1, w2, b2], name='Hypernet')

def train_step_drqn(drqn, drqn_target, optimizer, state_seq, actions_seq, rewards, h0, step_dims):
    rewards = tf.cast(rewards, tf.float32)
    with tf.GradientTape() as tape:
        *q_list, h_new = drqn([state_seq, h0])
        *q_target_list, _ = drqn_target([state_seq, h0])
        loss = 0.0
        for t, dim in enumerate(step_dims):
            q_pred = tf.reduce_sum(tf.one_hot(actions_seq[:,t], dim) * q_list[t], axis=1)
            if t < len(step_dims)-1:
                next_max = tf.reduce_max(q_target_list[t+1], axis=1)
                target = GAMMA * next_max
            else:
                target = rewards
            loss += compute_loss(q_pred,target)
        loss /= float(len(step_dims))
    grads = tape.gradient(loss, drqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, drqn.trainable_variables))
    return loss, h_new

def train_step_hyper(hyper, optimizer, states, actions, q4, q5, rewards):
    rewards = tf.cast(rewards, tf.float32)
    ohs = [tf.one_hot(actions[:,i], dim) for i, dim in enumerate(RE_ACTION_SPACE)]
    oh5 = tf.one_hot(actions[:,-1], PRE_ACTION_SPACE[0])
    inp = tf.concat([states] + ohs + [oh5], axis=1)
    with tf.GradientTape() as tape:
        w1_flat, b1, w2_flat, b2 = hyper(inp)
        B = tf.shape(states)[0]
        w1 = tf.reshape(w1_flat, [B, 2, EMBED_DIM])
        b1 = tf.reshape(b1, [B, EMBED_DIM])
        w2 = tf.reshape(w2_flat, [B, EMBED_DIM, 1])
        b2 = tf.reshape(b2, [B, 1])
        q_cat = tf.concat([q4, q5], axis=1)[:,None,:]  # [B,1,2]  # [B,1,2]
        hidden = tf.nn.relu(tf.matmul(q_cat, w1) + b1[:,None,:])
        q_tot = tf.matmul(hidden, w2) + b2[:,None]
        q_tot = tf.squeeze(q_tot, [1,2])
        loss = compute_loss(q_tot,rewards)
    grads = tape.gradient(loss, hyper.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, hyper.trainable_variables))
    return loss

def build_seq_remdqn():
    s = layers.Input((STATE_SPACE,), name='state')
    if USE_BATCH_NORM:
        s = tf.keras.layers.BatchNormalization()(s)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION,kernel_initializer=WEIGHT_INITIALIZER)(s)
    if DROPOUT_RATE > 0:
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION)(x)
    total_dim = SINGLE_AGENT_ACTION_DIM * ENSEMBLE_SIZE
    q_flat = layers.Dense(total_dim)(x)
    q_heads = layers.Reshape((SINGLE_AGENT_ACTION_DIM, ENSEMBLE_SIZE))(q_flat)
    return models.Model(s, q_heads, name='SeqREMDQN')

@tf.function
def train_step_seq_remdqn(dqn, dqn_target, optimizer, states, actions_seq, rewards):
    rewards = tf.cast(rewards, tf.float32)
    B = tf.shape(states)[0]
    w = tf.random.uniform((B, ENSEMBLE_SIZE), dtype=tf.float32)
    w = w / tf.reduce_sum(w, axis=1, keepdims=True)
    with tf.GradientTape() as tape:
        q_heads = dqn(states)               # [B, total, E]
        q_target_heads = dqn_target(states) # [B, total, E]
        w_exp = tf.expand_dims(w, axis=1)       # [B,1,E]
        q_w = tf.reduce_sum(q_heads * w_exp, axis=2)      # [B, total]
        q_t_w = tf.reduce_sum(q_target_heads * w_exp, axis=2)  # [B, total] tf.tensordot(q_heads, w, axes=[[2],[1]])
        q_t_w = tf.tensordot(q_target_heads, w, axes=[[2],[1]])
        dims = RE_ACTION_SPACE + PRE_ACTION_SPACE
        offset = 0
        loss = tf.constant(0.0, tf.float32)
        for t, dim in enumerate(dims):
            q_t = q_w[:, offset:offset+dim]
            q_pred = tf.reduce_sum(tf.one_hot(actions_seq[:,t], dim) * q_t, axis=1)
            if t < len(RE_ACTION_SPACE):
                next_q = q_t_w[:, offset+dim:offset+dim+dims[t+1]]
                next_max = tf.reduce_max(next_q, axis=1)
                target = GAMMA * next_max
            else:
                target = rewards
            loss += compute_loss(q_pred,target)
            offset += dim
        loss /= tf.cast(len(dims), tf.float32)
    grads = tape.gradient(loss, dqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
    return loss

def build_seq_emdqn():
    s = layers.Input((STATE_SPACE,), name='state')
    if USE_BATCH_NORM:
        s = tf.keras.layers.BatchNormalization()(s)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION,kernel_initializer=WEIGHT_INITIALIZER)(s)
    if DROPOUT_RATE > 0:
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION)(x)
    total_dim = SINGLE_AGENT_ACTION_DIM * ENSEMBLE_SIZE
    q_flat = layers.Dense(total_dim, activation=None)(x)
    q_heads = layers.Reshape((SINGLE_AGENT_ACTION_DIM, ENSEMBLE_SIZE))(q_flat)
    q_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2), name='q_mean')(q_heads)
    return models.Model(s, [q_heads, q_mean], name='SeqEMDQN')

@tf.function
def train_step_seq_emdqn(dqn, dqn_target, optimizer, states, actions_seq, rewards):
    rewards = tf.cast(rewards, tf.float32)
    with tf.GradientTape() as tape:
        q_heads, q_mean = dqn(states)     # heads: [B, total, E]; mean: [B, total]
        qt_heads_tgt, qt_mean_tgt = dqn_target(states)
        loss = tf.constant(0.0, tf.float32)
        dims = RE_ACTION_SPACE + PRE_ACTION_SPACE
        offset = 0
        for t, dim in enumerate(dims):
            q_t = q_mean[:, offset:offset+dim]  # [B, dim]
            q_pred = tf.reduce_sum(tf.one_hot(actions_seq[:,t], dim) * q_t, axis=1)
            if t < len(RE_ACTION_SPACE):
                next_q_tgt = qt_mean_tgt[:, offset+dim:offset+dim + dims[t+1]]
                next_max = tf.reduce_max(next_q_tgt, axis=1)
                target = GAMMA * next_max
            else:
                target = rewards
            loss += compute_loss(q_pred,target)
            offset += dim
        loss /= tf.cast(len(dims), tf.float32)
    grads = tape.gradient(loss, dqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
    return loss

def build_awrq(step_dims):
    state_in = layers.Input((None, STATE_SPACE), name='seq_state')
    h_in = layers.Input((HIDDEN_SIZE,), name='h0')
    rnn = layers.RNN(layers.GRUCell(HIDDEN_SIZE), return_sequences=True, return_state=True)
    seq_out, h_out = rnn(state_in, initial_state=[h_in])
    qs = []
    for t, dim in enumerate(step_dims):
        x = layers.Lambda(lambda x, i=t: x[:, i, :])(seq_out)
        if USE_BATCH_NORM:
            x = tf.keras.layers.BatchNormalization()(x)
        if DROPOUT_RATE > 0:
            x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        q_flat = layers.Dense(dim * ENSEMBLE_SIZE)(x)
        qs.append(layers.Reshape((dim, ENSEMBLE_SIZE))(q_flat))
    return models.Model([state_in, h_in], [*qs, h_out], name=f'DRQN_{len(step_dims)}')

def build_hypernet_awrq():
    tot_act = sum(RE_ACTION_SPACE) + sum(PRE_ACTION_SPACE)
    inp = layers.Input((STATE_SPACE + tot_act,), name='hyper_in')
    h = layers.Dense(128, activation=ACTIVATION_FUNCTION,kernel_initializer=WEIGHT_INITIALIZER)(inp)
    w1 = layers.Dense(EMBED_DIM * 2, activation='softplus', name='hyper_w1')(h)
    b1 = layers.Dense(EMBED_DIM, name='hyper_b1')(h)
    w2 = layers.Dense(EMBED_DIM, activation='softplus', name='hyper_w2')(h)
    b2 = layers.Dense(1, name='hyper_b2')(h)
    return models.Model(inp, [w1, b1, w2, b2], name='Hypernet')

def compute_vgca(actions_seq, rewards_batch, step_dims):
    label = tf.cast(tf.reshape(rewards_batch, [-1]), tf.float32)
    var_list = []
    for t in range(len(step_dims)):
        ua, idx = tf.unique(actions_seq[:, t])
        seg = tf.math.unsorted_segment_mean(label, idx, tf.shape(ua)[0])
        _, var = tf.nn.moments(seg, axes=[0])
        var_list.append(var)
    var_sum = tf.add_n(var_list)
    cum = tf.zeros([], dtype=tf.float32)
    weights = []
    for v in var_list:
        cum = cum + v
        w = cum / var_sum
        weights.append(w)
    return weights

def train_step_awrq(drqn, drqn_target, optimizer,
                     state_seq, actions_seq, rewards_batch, h0, step_dims):
    rewards_batch = tf.cast(rewards_batch, tf.float32)
    weights = compute_vgca(actions_seq, rewards_batch, step_dims)
    with tf.GradientTape() as tape:
        *q_list, h_new = drqn([state_seq, h0])
        if drqn_target is not None:
            *q_target, _ = drqn_target([state_seq, h0])
        head_losses = []
        for e in range(ENSEMBLE_SIZE):
            loss_e = tf.constant(0.0)
            for t in range(len(step_dims)):
                qt = q_list[t][:,:,e]
                if drqn_target is not None and t < len(step_dims)-1:
                    qt_n = q_target[t+1][:,:,e]
                    target = rewards_batch[:,None] * weights[t] + GAMMA * tf.reduce_max(qt_n,axis=1,keepdims=True)
                else:
                    target = rewards_batch[:,None] * weights[t]
                a = actions_seq[:, t]
                q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt,
                                       axis=1, keepdims=True)
                loss_e += compute_loss(q_pred,target)
            head_losses.append(loss_e)
        loss_stack = tf.stack(head_losses)
        en_w = loss_stack / tf.reduce_sum(loss_stack)
        ens_loss = tf.constant(0.0)
        for t in range(len(step_dims)):
            qt_stack = q_list[t]
            qt_e = tf.tensordot(qt_stack, en_w, axes=[[2],[0]])
            a = actions_seq[:, t]
            q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt_e,
                                   axis=1, keepdims=True)
            if drqn_target is not None and t < len(step_dims)-1:
                qt_n_stack = q_target[t+1]
                qt_n_e = tf.tensordot(qt_n_stack, en_w, axes=[[2],[0]])
                target = rewards_batch[:,None] * weights[t] + GAMMA * tf.reduce_max(qt_n_e,axis=1,keepdims=True)
            else:
                target = rewards_batch[:,None] * weights[t]
            ens_loss += compute_loss(q_pred,target)
        total_loss = (tf.reduce_sum(head_losses) + ens_loss) / (len(step_dims) * (ENSEMBLE_SIZE+1))
    grads = tape.gradient(total_loss, drqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, drqn.trainable_variables))
    qt_last = q_list[-1]
    qt_e = tf.tensordot(qt_last, en_w, axes=[[2],[0]])
    q_sel = tf.reduce_sum(tf.one_hot(actions_seq[:,-1], step_dims[-1]) * qt_e,
                            axis=1, keepdims=True)
    return total_loss, h_new, q_sel

def train_step_awrq_vgca(drqn, drqn_target, optimizer,
                     state_seq, actions_seq, rewards_batch, h0, step_dims):
    rewards_batch = tf.cast(rewards_batch, tf.float32)
    with tf.GradientTape() as tape:
        *q_list, h_new = drqn([state_seq, h0])
        if drqn_target is not None:
            *q_target, _ = drqn_target([state_seq, h0])
        head_losses = []
        for e in range(ENSEMBLE_SIZE):
            loss_e = tf.constant(0.0)
            for t in range(len(step_dims)):
                qt = q_list[t][:,:,e]
                if drqn_target is not None and t < len(step_dims)-1:
                    qt_n = q_target[t+1][:,:,e]
                    target = GAMMA * tf.reduce_max(qt_n,axis=1,keepdims=True)
                else:
                    target = rewards_batch[:,None]
                a = actions_seq[:, t]
                q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt,
                                       axis=1, keepdims=True)
                loss_e += compute_loss(q_pred,target)
            head_losses.append(loss_e)
        loss_stack = tf.stack(head_losses)
        en_w = loss_stack / tf.reduce_sum(loss_stack)
        ens_loss = tf.constant(0.0)
        for t in range(len(step_dims)):
            qt_stack = q_list[t]
            qt_e = tf.tensordot(qt_stack, en_w, axes=[[2],[0]])
            a = actions_seq[:, t]
            q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt_e,
                                   axis=1, keepdims=True)
            if drqn_target is not None and t < len(step_dims)-1:
                qt_n_stack = q_target[t+1]
                qt_n_e = tf.tensordot(qt_n_stack, en_w, axes=[[2],[0]])
                target = GAMMA * tf.reduce_max(qt_n_e,axis=1,keepdims=True)
            else:
                target = rewards_batch[:,None] 
            ens_loss += compute_loss(q_pred,target)
        total_loss = (tf.reduce_sum(head_losses) + ens_loss) / (len(step_dims) * (ENSEMBLE_SIZE+1))
    grads = tape.gradient(total_loss, drqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, drqn.trainable_variables))
    qt_last = q_list[-1]
    qt_e = tf.tensordot(qt_last, en_w, axes=[[2],[0]])
    q_sel = tf.reduce_sum(tf.one_hot(actions_seq[:,-1], step_dims[-1]) * qt_e,
                            axis=1, keepdims=True)
    return total_loss, h_new, q_sel

def train_step_awrq_aw(drqn, drqn_target, optimizer,
                     state_seq, actions_seq, rewards_batch, h0, step_dims):
    rewards_batch = tf.cast(rewards_batch, tf.float32)
    weights = compute_vgca(actions_seq, rewards_batch, step_dims)
    with tf.GradientTape() as tape:
        *q_list, h_new = drqn([state_seq, h0])
        if drqn_target is not None:
            *q_target, _ = drqn_target([state_seq, h0])
        head_losses = []
        for e in range(ENSEMBLE_SIZE):
            loss_e = tf.constant(0.0)
            for t in range(len(step_dims)):
                qt = q_list[t][:,:,e]
                if drqn_target is not None and t < len(step_dims)-1:
                    qt_n = q_target[t+1][:,:,e]
                    target = rewards_batch[:,None] * weights[t] + GAMMA * tf.reduce_max(qt_n,axis=1,keepdims=True)
                else:
                    target = rewards_batch[:,None] * weights[t]
                a = actions_seq[:, t]
                q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt,
                                       axis=1, keepdims=True)
                loss_e += compute_loss(q_pred,target)
            head_losses.append(loss_e)
        rand_w = tf.random.uniform((ENSEMBLE_SIZE,), dtype=tf.float32)
        en_w = rand_w / tf.reduce_sum(rand_w)
        ens_loss = tf.constant(0.0)
        for t in range(len(step_dims)):
            qt_stack = q_list[t]
            qt_e = tf.tensordot(qt_stack, en_w, axes=[[2],[0]])
            a = actions_seq[:, t]
            q_pred = tf.reduce_sum(tf.one_hot(a, step_dims[t]) * qt_e,
                                   axis=1, keepdims=True)
            if drqn_target is not None and t < len(step_dims)-1:
                qt_n_stack = q_target[t+1]
                qt_n_e = tf.tensordot(qt_n_stack, en_w, axes=[[2],[0]])
                target = rewards_batch[:,None] * weights[t] + GAMMA * tf.reduce_max(qt_n_e,axis=1,keepdims=True)
            else:
                target = rewards_batch[:,None] * weights[t]
            ens_loss += compute_loss(q_pred,target)
        total_loss = (tf.reduce_sum(head_losses) + ens_loss) / (len(step_dims) * (ENSEMBLE_SIZE+1))
    grads = tape.gradient(total_loss, drqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, drqn.trainable_variables))
    qt_last = q_list[-1]
    qt_e = tf.tensordot(qt_last, en_w, axes=[[2],[0]])
    q_sel = tf.reduce_sum(tf.one_hot(actions_seq[:,-1], step_dims[-1]) * qt_e,
                            axis=1, keepdims=True)
    return total_loss, h_new, q_sel

def train_step_hyper_awrq(hypernet, optimizer, states, actions, q4_sel, q5_sel, rewards):
    states = tf.cast(states, tf.float32)
    rewards = tf.cast(rewards, tf.float32)
    
    with tf.GradientTape() as tape:
        ohs = [tf.one_hot(actions[:, t], dim) for t, dim in enumerate(RE_ACTION_SPACE)]
        oh5 = tf.one_hot(actions[:, -1], PRE_ACTION_SPACE[0])
        hyper_in = tf.concat([states] + ohs + [oh5], axis=1)

        w1_flat, b1, w2_flat, b2 = hypernet(hyper_in)
        B = tf.shape(states)[0]
        w1 = tf.reshape(w1_flat, [B, 2, EMBED_DIM])
        b1 = tf.reshape(b1, [B, EMBED_DIM])
        w2 = tf.reshape(w2_flat, [B, EMBED_DIM, 1])
        b2 = tf.reshape(b2, [B, 1])

        q4 = tf.reshape(q4_sel, [B, 1, 1])
        q5 = tf.reshape(q5_sel, [B, 1, 1])
        q_cat = tf.concat([q4, q5], axis=2)  # [B,1,2]

        hidden = tf.nn.relu(tf.matmul(q_cat, w1) + b1[:, None, :])  # [B,1,embed]
        q_tot = tf.matmul(hidden, w2) + b2[:, None]                # [B,1,1]
        q_tot = tf.squeeze(q_tot, axis=[1, 2])                     # [B]
        
        loss = compute_loss(q_tot,rewards)

    grads = tape.gradient(loss, hypernet.trainable_variables)
    grad_norms = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(g))) for g in grads]) 
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, hypernet.trainable_variables))

    return loss, grad_norms

def build_seq_dqn():
    s = layers.Input((STATE_SPACE,), name='state')
    if USE_BATCH_NORM:
        s = tf.keras.layers.BatchNormalization()(s)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION,kernel_initializer=WEIGHT_INITIALIZER)(s)
    if DROPOUT_RATE > 0:
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation=ACTIVATION_FUNCTION)(x)
    total_dim = sum(RE_ACTION_SPACE) + sum(PRE_ACTION_SPACE)
    q_flat = layers.Dense(total_dim, activation=None)(x)
    outputs = []
    idx = 0
    for dim in RE_ACTION_SPACE:
        outputs.append(layers.Lambda(lambda x, start, d: x[:, start:start+d],
                                     arguments={'start': idx, 'd': dim})(q_flat))
        idx += dim
    outputs.append(layers.Lambda(lambda x, start, d: x[:, start:start+d],
                                 arguments={'start': idx, 'd': PRE_ACTION_SPACE[0]})(q_flat))
    return models.Model(s, outputs, name='SeqDQN')

@tf.function
def train_step_seq_dqn(dqn, dqn_target, optimizer, states, actions_seq, rewards):
    rewards = tf.cast(rewards, tf.float32)
    with tf.GradientTape() as tape:
        q_list = dqn(states)               # list of [B, dim_t]
        q_target_list = dqn_target(states) # target network
        loss = tf.constant(0.0, tf.float32)
        for t, dim in enumerate(RE_ACTION_SPACE + PRE_ACTION_SPACE):
            q_pred = tf.reduce_sum(
                tf.one_hot(actions_seq[:,t], dim) * q_list[t], axis=1, keepdims=True)
            if t < len(RE_ACTION_SPACE):
                next_max = tf.reduce_max(q_target_list[t+1], axis=1, keepdims=True)
                target = GAMMA * next_max
            else:
                target = rewards[:,None]
            loss += compute_loss(q_pred,target)
        loss = loss / (len(RE_ACTION_SPACE) + 1)
    grads = tape.gradient(loss, dqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
    return loss

@tf.function
def train_step_double_dqn(dqn, dqn_target, optimizer, states, actions_seq, rewards):
    rewards = tf.cast(rewards, tf.float32)
    dims_list = RE_ACTION_SPACE + PRE_ACTION_SPACE
    with tf.GradientTape() as tape:
        q_list = dqn(states)
        q_target_list = dqn_target(states)
        loss = tf.constant(0.0, tf.float32)
        for t in range(len(dims_list)):
            dim = dims_list[t]
            q_pred = tf.reduce_sum(tf.one_hot(actions_seq[:,t], dim) * q_list[t], axis=1)
            if t < len(RE_ACTION_SPACE):
                next_q_online = q_list[t+1]
                next_action = tf.argmax(next_q_online, axis=1)
                next_q_target = q_target_list[t+1]
                next_dim = dims_list[t+1]
                next_max = tf.reduce_sum(tf.one_hot(next_action, next_dim) * next_q_target, axis=1)
                target = GAMMA * next_max
            else:
                target = rewards
            loss += compute_loss(q_pred,target)
        loss /= len(dims_list)
    grads = tape.gradient(loss, dqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
    return loss

def build_seq_drqn():
    state_seq = layers.Input((None, STATE_SPACE), name='state_seq')  # [B, T, D]
    h0 = layers.Input((HIDDEN_SIZE,), name='h0')
    rnn = layers.RNN(layers.GRUCell(HIDDEN_SIZE), return_sequences=True)
    seq_out = rnn(state_seq, initial_state=[h0])                 # [B, T, H]
    qs = []
    for t, dim in enumerate(RE_ACTION_SPACE + PRE_ACTION_SPACE):
        x = layers.Lambda(lambda x, i=t: x[:, i, :])(seq_out)
        if USE_BATCH_NORM:
            x = tf.keras.layers.BatchNormalization()(x)
        if DROPOUT_RATE > 0:
            x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        q = layers.Dense(dim, activation=None)(x)
        qs.append(q)
    return models.Model([state_seq, h0], qs, name='SeqDRQN')

@tf.function
def train_step_seq_drqn(drqn, drqn_target, optimizer,
                          state_seq, actions_seq, rewards):
    rewards = tf.cast(rewards, tf.float32)
    B = tf.shape(state_seq)[0]
    h0 = tf.zeros((B, HIDDEN_SIZE), tf.float32)

    with tf.GradientTape() as tape:
        q_list = drqn([state_seq, h0])
        q_target_list = drqn_target([state_seq, h0]) if drqn_target is not None else None

        loss = tf.constant(0.0, tf.float32)
        dims = RE_ACTION_SPACE + PRE_ACTION_SPACE
        for t, dim in enumerate(dims):
            q_pred = tf.reduce_sum(
                tf.one_hot(actions_seq[:, t], dim) * q_list[t], axis=1
            )  # [B]

            if t < len(RE_ACTION_SPACE):
                next_max = tf.reduce_max(q_target_list[t+1], axis=1)
                target = GAMMA * next_max
            else:
                target = rewards

            loss += compute_loss(q_pred,target)
        loss /= tf.cast(len(dims), tf.float32)
    grads = tape.gradient(loss, drqn.trainable_variables)
    grads = clip_gradients(optimizer, grads)
    optimizer.apply_gradients(zip(grads, drqn.trainable_variables))

    return loss