import tensorflow as tf


def CAB(inpt, filters, k):
    hdn = tf.keras.layers.Conv2D(filters, k, padding='same', activation=None,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01))(inpt)
    hdn = tf.keras.layers.LeakyReLU(alpha=0.2)(hdn)
    hdn = tf.keras.layers.BatchNormalization()(hdn)
    return hdn


def model1(image_size, n_attributes, label_frequencies):

    inpt = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    hdn = CAB(inpt, 8, 3)
    hdn = tf.keras.layers.MaxPooling2D()(hdn)

    hdn = CAB(hdn, 16, 3)
    hdn = tf.keras.layers.MaxPooling2D()(hdn)

    hdn = CAB(hdn, 32, 3)
    hdn = tf.keras.layers.MaxPooling2D()(hdn)

    hdn = CAB(hdn, 64, 3)
    hdn = tf.keras.layers.MaxPooling2D()(hdn)

    hdn = tf.keras.layers.Flatten()(hdn)

    hdn = tf.keras.layers.Dense(512, activation=None,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01))(hdn)
    hdn = tf.keras.layers.LeakyReLU(alpha=0.2)(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.4)(hdn)

    hdn = tf.keras.layers.Dense(128, activation=None,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01))(hdn)
    hdn = tf.keras.layers.LeakyReLU(alpha=0.2)(hdn)
    hdn = tf.keras.layers.Dropout(rate=0.4)(hdn)

    hdn = tf.keras.layers.Dense(n_attributes, activation='sigmoid')(hdn)

    model = tf.keras.models.Model(inputs=inpt, outputs=hdn)
    opt = tf.keras.optimizers.Adam()

    if label_frequencies is not None:
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryCrossentropy()],
                      loss_weights=label_frequencies)
    else:
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryCrossentropy()])

    return model