import tensorflow as tf

class SpamLDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, optimizer=None, loss=None, metrics=None, model_path=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics

        if model_path:
            self.model = self.__load_model(model_path)
        else:
            self.model = self.__build_model()
            self.__compile_model()

    def __build_model(self):
        """
        Private method untuk membangun arsitektur model.
        """
        input_layer = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(12, activation='relu')(input_layer)
        output_layer = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)

        # Create the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    def __compile_model(self):
        """
        Private method untuk mengkompilasi model.
        """
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

    def __load_model(self, model_path):
        """
        Private method untuk memuat model yang sudah dilatih dari file.
        """
        return tf.keras.models.load_model(model_path)
    
    def train(self, train_data, validation_data, epochs=10, callbacks=None, **kwargs):
        """
        Metode untuk melatih model dengan data yang diberikan.
        """
        history = self.model.fit(train_data,
                                 validation_data=validation_data,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 **kwargs)
        return history