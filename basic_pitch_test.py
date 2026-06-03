from basic_pitch.inference import predict_and_save
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

try:
    # Attempt a dry load of the model
    from basic_pitch.inference import Model
    print("Model class successfully imported.")
except Exception as e:
    print(f"Model import failed: {e}")