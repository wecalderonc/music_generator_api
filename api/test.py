import joblib
import tensorflow as tf
import h5py

MODEL_NAME = "music-generator-model"
BUCKET_NAME = "music-generator-713"
MODEL_VERSION = "0.1"
LOCAL_MODEL_NAME = "model.h5"
BUCKET_NAME = 'music-generator-713'
BUCKET_TRAIN_DATA_PATH = 'data/'

gcs_path = f"gs://{BUCKET_NAME}/models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
print(gcs_path)
loaded_model = tf.io.gfile.GFile(gcs_path, 'rb')
model_gcs = h5py.File(loaded_model, 'r')
print(model_gcs)

