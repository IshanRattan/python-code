
from nmt_transformer.build_data import build_training_data
from nmt_transformer.transformer import Transformer
from nmt_transformer.configs import config

import tensorflow as tf
import time


tf.keras.backend.clear_session()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

if __name__ == '__main__':
    dataset, VOCAB_SIZE_EN, VOCAB_SIZE_FR = build_training_data()

    # Hyper-parameters
    D_MODEL = 128 # 512
    NB_LAYERS = 4 # 6
    FFN_UNITS = 512 # 2048
    NB_PROJ = 8 # 8, Attention heads
    DROPOUT_RATE = 0.1 # 0.1

    transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                              vocab_size_dec=VOCAB_SIZE_FR,
                              d_model=D_MODEL,
                              nb_layers=NB_LAYERS,
                              FFN_units=FFN_UNITS,
                              nb_proj=NB_PROJ,
                              dropout=DROPOUT_RATE)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction="none")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, config.checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    EPOCHS = 5
    for epoch in range(EPOCHS):
        print("Start of epoch {}".format(epoch + 1))
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            dec_inputs = targets[:, :-1] # This op removed the last column of the target language tensor
            dec_outputs_real = targets[:, 1:] # This op removed the first column of the tensor language tensor
            with tf.GradientTape() as tape:
                predictions = transformer(enc_inputs, dec_inputs, True)
                loss = loss_function(dec_outputs_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(dec_outputs_real, predictions)

            if batch % 50 == 0:
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} at {}".format(epoch + 1, ckpt_save_path))
        print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))