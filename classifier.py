import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 32
seed = 42

# 80% of test data for training
raw_train_ds = tf.keras.utils.text_dataset_from_directory('struc', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)

# 20% of test data for validation, not always needed
raw_val_ds = tf.keras.utils.text_dataset_from_directory('struc', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)

# 100% of test data for training
raw_test_ds = tf.keras.utils.text_dataset_from_directory('struc', batch_size=batch_size)


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# plt.plot(epochs, loss, 'go', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.plot(epochs, acc, 'go', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')

# plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# print(export_model.predict(
#   [
#     "My sock is twisted.", 
#     "I'm so happy I passed my exam, I could die.",
#     "I hate life.",
#     "I can't take it anymore, I'm going to hang myself."
#   ]
# ))

# export_model.save('suicide_classifier', save_format='tf')