import shutil
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.layers as layers


class GAN:
    def __init__(self, train_df, noise_dim, gen_lr, dis_lr, batch_size, epochs, seed=1234):
        self.data_dim = train_df.shape[1]
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_number_of_steps = epochs
        self.train_df = train_df

        self.queues = tf.contrib.slim.queues
        self.ds = tf.contrib.distributions
        self.framework = tf.contrib.framework

        # Prevent to use previous training data
        # shutil.rmtree('./output/tf')

        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.reset_default_graph()

        # Optimizer for GAN networks
        generator_optimizer = tf.compat.v1.train.AdamOptimizer(gen_lr, beta1=0.5)
        discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(dis_lr, beta1=0.5)

        # Initialize GANEstimator with options and hyper-parameters.
        self.gan_estimator = tfgan.estimator.GANEstimator(
            generator_fn=self.generator_fn,
            discriminator_fn=self.discriminator_fn,

            # Vanilla GAN loss
            # generator_loss_fn=tfgan.losses.minimax_generator_loss,
            # discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,

            # Wasserstein loss (https://arxiv.org/abs/1701.07875) with the gradient penalty
            # from the improved Wasserstein loss paper (https://arxiv.org/abs/1704.00028).
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,

            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,

            add_summaries=tfgan.estimator.SummaryType.VARIABLES,
            model_dir='./output/tf')

    def _get_train_input_fn(self, batch_size, noise_dims):
        def train_input_fn():
            noise = tf.random.normal([batch_size, noise_dims])
            train_dataset = self.convert_to_dataset(self.train_df, batch_size)
            return noise, train_dataset
        return train_input_fn

    @staticmethod
    def _get_predict_input_fn(gen_size, noise_dims):
        def predict_input_fn():
            return tf.random.normal([gen_size, noise_dims])
        return predict_input_fn

    def train(self):
        # Train estimator
        train_input_fn = self._get_train_input_fn(self.batch_size, self.noise_dim)
        self.gan_estimator.train(train_input_fn, steps=self.epochs)

    def predict(self, gen_size):
        # For disable below warning
        # WARNING:tensorflow:Input graph does not use tf.data.Dataset or contain a QueueRunner.
        # That means predict yields forever. This is probably a mistake.
        tf.estimator.Estimator._validate_features_in_predict_input = lambda *args: None

        # Run inference
        def _get_next(iterable):
            return iterable.__next__()

        predict_input_fn = self._get_predict_input_fn(gen_size, self.noise_dim)
        prediction_iterable = self.gan_estimator.predict(predict_input_fn)
        predictions = [_get_next(prediction_iterable) for _ in range(gen_size)]

        try:  # Close the predict session.
            _get_next(prediction_iterable)
        except StopIteration:
            pass

        return predictions

    def convert_to_dataset(self, train_df, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(train_df.astype('float32').values)
        batched_dataset = self.dataset_to_stream(dataset, batch_size)
        return batched_dataset

    @staticmethod
    def dataset_to_stream(inp, batch_size):
        with tf.device('/cpu:0'):
            batched = inp.batch(batch_size, drop_remainder=True)
            data_feeder = tf.compat.v1.data.make_one_shot_iterator(batched.repeat()).get_next()
        return data_feeder

    def generator_fn(self, noise, weight_decay=2.5e-5, is_training=True):
        return self._generator(noise, weight_decay, is_training)

    def _generator(self, noise, weight_decay, is_training):
        with tf.contrib.framework.arg_scope(
                [layers.fully_connected],
                activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                weights_regularizer=layers.l2_regularizer(weight_decay)):
            with tf.contrib.framework.arg_scope([layers.batch_norm], is_training=is_training):
                net = layers.fully_connected(noise, 64)
                net = layers.fully_connected(net, 64)
                net = layers.fully_connected(net, 64)
                net = layers.fully_connected(net, 32)
                net = layers.fully_connected(net, 32)
                net = layers.fully_connected(net, 32)
                net = layers.fully_connected(net, 16)
                net = layers.fully_connected(net, 16)
                net = layers.fully_connected(net, self.data_dim, activation_fn=tf.nn.sigmoid)
                return net

    def discriminator_fn(self, inputs, weight_decay=2.5e-5):
        return self._discriminator(inputs, weight_decay)

    @staticmethod
    def _discriminator(inputs, weight_decay):
        with tf.contrib.framework.arg_scope(
                [layers.fully_connected],
                activation_fn=tf.nn.leaky_relu, normalizer_fn=None,
                weights_regularizer=layers.l2_regularizer(weight_decay),
                biases_regularizer=layers.l2_regularizer(weight_decay)):
            net = layers.fully_connected(inputs, 64)
            net = layers.fully_connected(net, 64)
            net = layers.fully_connected(net, 64)
            net = layers.fully_connected(net, 32)
            net = layers.fully_connected(net, 32)
            net = layers.fully_connected(net, 32)
            net = layers.fully_connected(net, 16)
            net = layers.fully_connected(net, 16)
            net = layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=layers.layer_norm)
            return net
