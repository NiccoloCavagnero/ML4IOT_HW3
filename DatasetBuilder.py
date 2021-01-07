import tensorflow as tf
import os 

class SignalGenerator:

  def __init__(self, keywords, samp_rate, frame_length, frame_step, num_bins = None,
               lower_frequency=None, upper_frequency=None,
              num_coefficients=None, mfccs=False):
    self.keywords = keywords
    self.samp_rate = samp_rate
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.num_bins = num_bins
    self.lower_frequency = lower_frequency
    self.upper_frequency = upper_frequency   
    self.num_coefficients = num_coefficients
    num_spectrogram_bins = (frame_length) // 2 + 1

    if mfccs:
      self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_bins, num_spectrogram_bins, self.samp_rate,
                    self.lower_frequency, self.upper_frequency)
      self.preprocess = self.preprocess_mfccs
    else:
      self.preprocess = self.preprocess_stft

  def read(self, path):
    parts = tf.strings.split(path, os.path.sep)
    label = parts[-2]
    label_id = tf.argmax(label == self.keywords)
    audio_binary = tf.io.read_file(path)
    audio,_ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)

    return audio,label_id

  def padding(self,audio):
    zero_padding = tf.zeros([self.samp_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([self.samp_rate])

    return audio

  def get_spectrogram(self,audio):
    stft = tf.signal.stft(audio, frame_length=self.frame_length, 
                          frame_step = self.frame_step, fft_length = self.frame_length)
    spectrogram = tf.abs(stft)

    return spectrogram
  
  def get_mfccs(self, spectrogram):
    mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :self.num_coefficients]

    return mfccs    

  def preprocess_stft(self,path):
    audio,label = self.read(path)
    audio = self.padding(audio)
    spectrogram = self.get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [32, 32])

    return spectrogram, label

  def preprocess_mfccs(self,path):
    audio,label = self.read(path)
    audio = self.padding(audio)
    spectrogram = self.get_spectrogram(audio)
    mfccs = self.get_mfccs(spectrogram)
    mfccs = tf.expand_dims(mfccs, -1)

    return mfccs, label
  
  def make_dataset(self, files, train):
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(self.preprocess, num_parallel_calls=4)
    ds = ds.batch(32)
    ds = ds.cache()
    if train:
      ds = ds.shuffle(100, reshuffle_each_iteration=True)

    return ds
