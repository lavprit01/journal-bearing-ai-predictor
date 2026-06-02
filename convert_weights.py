# convert_weights.py
import numpy as np
import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model('models/ffnn_model.keras')

weights = model.get_weights()
weights_array = np.empty(len(weights), dtype=object)
weights_array[:] = weights

np.save('models/ffnn_weights.npy', weights_array, allow_pickle=True)
print("✅ Successfully extracted and saved weights to models/ffnn_weights.npy")