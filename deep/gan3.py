import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.gan as tfgan


class GAN:
    def __init__(self, train_df, noise_dim, gen_lr, dis_lr, batch_size, epochs, seed=1234):
        self.data_dim = train_df.shape[1]
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_number_of_steps = epochs

        self.queues = tf.contrib.slim.queues
        self.layers = tf.contrib.layers
        self.ds = tf.contrib.distributions
        self.framework = tf.contrib.framework

        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.reset_default_graph()

        self.train_dataset = self.convert_to_dataset(train_df, batch_size)
        self.gan_model = self.build_gan_model()
        self.gan_loss = self.build_gan_loss(self.gan_model)
        self.gan_ops = self.build_train_optimizer(self.gan_model, self.gan_loss, gen_lr, dis_lr)
        self.evaluate_tfgan_loss(self.gan_loss)

    def build_gan_model(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        gan_model = tfgan.gan_model(
            generator_fn=self.generator,
            discriminator_fn=self.discriminator,
            real_data=self.train_dataset,
            generator_inputs=noise)

        return gan_model

    @staticmethod
    def build_gan_loss(gan_model):
        # Example of classical loss function.
        # vanilla_gan_loss = tfgan.gan_loss(
        #    gan_model,
        #    generator_loss_fn=tfgan.losses.minimax_generator_loss,
        #    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)
        # return vanilla_gan_loss

        # Wasserstein loss (https://arxiv.org/abs/1701.07875) with the
        # gradient penalty from the improved Wasserstein loss paper
        # (https://arxiv.org/abs/1704.00028).
        improved_wgan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            gradient_penalty_weight=1.0)

        return improved_wgan_loss

    @staticmethod
    def build_train_optimizer(gan_model, gan_loss, gen_lr, dis_lr):
        generator_optimizer = tf.compat.v1.train.AdamOptimizer(gen_lr, beta1=0.5)
        discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(dis_lr, beta1=0.5)
        gan_train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            summarize_gradients=True,)

        return gan_train_ops

    def train(self):
        # tfgan.gan_train(
        #     self.gan_ops,
        #     hooks=[tf.train.StopAtStepHook(num_steps=self.max_number_of_steps)],
        #     logdir='./output/tf')
        train_step_fn = tfgan.get_sequential_train_steps()
        global_step = tf.train.get_or_create_global_step()
        loss_values = []

        with tf.train.SingularMonitoredSession() as sess:
            for i in range(1601):
                cur_loss, _ = train_step_fn(sess, self.gan_ops, global_step, train_step_kwargs={})
                loss_values.append((i, cur_loss))
                if i % 200 == 0:
                    gen_data = sess.run([self.predict])
                    print(gen_data)
                    print(loss_values)

    def predict(self, gen_size):
        with tf.variable_scope('Generator', reuse=True):
            noise = tf.random.normal([gen_size, self.noise_dim])
            gen_data = self.gan_model.generator_fn(noise)
        return gen_data.eval()

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

    def evaluate_tfgan_loss(self, gan_loss, name=None):
        """Evaluate GAN losses. Used to check that the graph is correct.

        Args:
            gan_loss: A GANLoss tuple.
            name: Optional. If present, append to debug output.
        """
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            with self.queues.QueueRunners(sess):
                gen_loss_np = sess.run(gan_loss.generator_loss)
                dis_loss_np = sess.run(gan_loss.discriminator_loss)
        if name:
            print('%s generator loss: %f' % (name, gen_loss_np))
            print('%s discriminator loss: %f' % (name, dis_loss_np))
        else:
            print('Generator loss: %f' % gen_loss_np)
            print('Discriminator loss: %f' % dis_loss_np)

    def generator(self, inputs):
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.noise_dim, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(self.data_dim, activation='sigmoid', kernel_initializer='he_normal'),
        ])(inputs)
        return model

    def discriminator(self, inputs, unused_conditioning):
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.data_dim, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu'),
            keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal'),
        ])(inputs)
        return model
