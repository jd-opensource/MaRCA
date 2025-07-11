import tensorflow as tf
from tensorflow.keras import layers, models


class DCN(layers.Layer):
    def __init__(self, hidden_units, cross_layers):
        super(DCN, self).__init__()
        self.hidden_units = hidden_units
        self.cross_layers = cross_layers
        self.cross = [layers.Dense(1, use_bias=False) for _ in range(cross_layers)]
        self.deep = [layers.Dense(units, activation='relu') for units in hidden_units]

    def call(self, inputs):
        x0 = inputs
        x = inputs
        for i in range(self.cross_layers):
            x = x0 * self.cross[i](x) + x
        for layer in self.deep:
            x = layer(x)
        return x


def build_mmoe_model(wide_dim, deep_dim, g1_dim, g2_dim, num_experts=4, num_tasks=2, expert_dim=32):
    wide_input = layers.Input(shape=(wide_dim,), name='wide_input')
    deep_input = layers.Input(shape=(deep_dim,), name='deep_input')
    g1_input = layers.Input(shape=(g1_dim,), name='g1_input')
    g2_input = layers.Input(shape=(g2_dim,), name='g2_input')

    wide_output = layers.Dense(16, activation='relu')(wide_input)

    dcn = DCN(hidden_units=[64, 32], cross_layers=3)
    deep_output = dcn(deep_input)

    combined = layers.Concatenate()([wide_output, deep_output])

    experts = [layers.Dense(expert_dim, activation='relu')(combined) for _ in range(num_experts)]
    experts_concat = layers.Concatenate()(experts)

    gate1 = layers.Dense(num_experts, activation='softmax')(g1_input)
    gate2 = layers.Dense(num_experts, activation='softmax')(g2_input)

    mmoe_output1 = layers.Dot(axes=(1, 1))([gate1, layers.Reshape((num_experts, expert_dim))(experts_concat)])
    mmoe_output2 = layers.Dot(axes=(1, 1))([gate2, layers.Reshape((num_experts, expert_dim))(experts_concat)])

    task1_output = layers.Dense(1, name='task1_output')(layers.Flatten()(mmoe_output1))
    task2_output = layers.Dense(1, name='task2_output')(layers.Flatten()(mmoe_output2))

    model = models.Model(inputs=[wide_input, deep_input, g1_input, g2_input],
                         outputs=[task1_output, task2_output])
    return model
