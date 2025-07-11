import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from config import STATE_DIM, ACTION_DIMS, LEARNING_RATE, BATCH_SIZE, EPOCHS, DATA_PATH
from model import build_mmoe_model
import csv


def hash_state(state):
    return [hash(val) if isinstance(val, str) else val for val in state]


def load_data(file):
    W, D, G1, G2, R = [], [], [], [], []
    with open(file) as f:
        for row in csv.reader(f):
            vals = []
            for i, val in enumerate(row):
                try:
                    vals.append(float(val))
                except ValueError:
                    vals.append(val)

            w = vals[:STATE_DIM + len(ACTION_DIMS) + 1]
            d = vals[:STATE_DIM + len(ACTION_DIMS) + 1]
            W.append(hash_state(w))
            D.append(hash_state(d))

            G1.append(list(map(int, vals[STATE_DIM:STATE_DIM + 3])))
            G2.append(list(map(int, vals[STATE_DIM:STATE_DIM + len(ACTION_DIMS) + 1])))
            R.append(vals[STATE_DIM + len(ACTION_DIMS) + 1:])

    W = np.array(W, np.float32)
    D = np.array(D, np.float32)
    G1 = np.array(G1, np.int32)
    G2 = np.array(G2, np.int32)
    R = np.array(R, np.float32)

    w_mean, w_std = W.mean(0), W.std(0) + 1e-6
    d_mean, d_std = D.mean(0), D.std(0) + 1e-6
    return (W - w_mean) / w_std, (D - d_mean) / d_std, G1, G2, R


def train_mmoe(model, W, D, G1, G2, R):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    mse_loss = tf.keras.losses.MeanSquaredError()

    N = len(W)
    batches = (N + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(batches):
            start = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, N)
            batch_w = W[start:end]
            batch_d = D[start:end]
            batch_g1 = G1[start:end]
            batch_g2 = G2[start:end]
            batch_r = R[start:end]

            with tf.GradientTape() as tape:
                predictions = model([batch_w, batch_d, batch_g1, batch_g2])
                loss1 = mse_loss(batch_r[:, 0], predictions[0])
                loss2 = mse_loss(batch_r[:, 1], predictions[1])
                total_batch_loss = loss1 + loss2

            grads = tape.gradient(total_batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += total_batch_loss.numpy()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / batches:.4f}")


def main():
    W, D, G1, G2, R = load_data(DATA_PATH)

    model = build_mmoe_model(W.shape[1], D.shape[1], G1.shape[1], G2.shape[1])

    train_mmoe(model, W, D, G1, G2, R)

    print("train finished")


if __name__ == "__main__":
    main()