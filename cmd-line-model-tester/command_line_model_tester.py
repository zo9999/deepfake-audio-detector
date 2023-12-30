import tensorflow as tf
import argparse
import librosa
import numpy as np

#parse arguments
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-m", "--model", dest="model_path", required=True, help="Keras model")
arg_parser.add_argument("-c", "--clips", dest="clips", nargs="+", required=True, help="Audio clips to classify")
arg_parser.add_argument("-n", "--n-mels", default=91, dest="n_mels", help="Number of mel bands in the spectogram")

args = arg_parser.parse_args()

#load clips
x = []
for clip in args.clips:
  audio, _ = librosa.load(clip)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_mels=args.n_mels)
  mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

  max_time_steps = 150
  if mel_spectrogram.shape[1] < max_time_steps:
      mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
  else:
      mel_spectrogram = mel_spectrogram[:, :max_time_steps]

  x.append(mel_spectrogram)

x = np.array(x)

#load keras model
model = tf.keras.models.load_model(args.model_path)

predictions = model.predict(x)

for p in predictions:
  print("prediction")
  print("chance of bonafide: " + str((p[1]*100)) )
  print("chance of spoof: " + str((p[0]*100)) )
print(predictions)

