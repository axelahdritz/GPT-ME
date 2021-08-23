import io
import os
import string
import pandas as pd
import numpy as np
import wave
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

bucketname = "machine_feed_bucket"

filepath = os.path.abspath("audio/Processed/") + "/"
output_filepath = os.path.abspath("Transcripts/") + "/"

word_filepath = os.path.abspath("Transcripts/word_transcripts/") + "/"
sentence_filepath = os.path.abspath("Transcripts/sentence_transcripts/") + "/"
transcript_filepath = os.path.abspath("Transcripts/full_transcripts/") + "/"

def get_date(audio_file_name):
    year = audio_file_name[:2]
    month = audio_file_name[2:4]
    day = audio_file_name[4:6]
    date = month + '/' + day + '/' + year
    
    hour = audio_file_name[7:9]
    minute = audio_file_name[9:11]
    ctime = hour + ':' + minute
    return date, ctime

def word_counter(words):
    count = 0
    tokens = words.split()
    for word in tokens:
        count += 1
    return count

def mp3_to_wav(audio_file_name):
    if audio_file_name.split('.')[1] == 'mp3':    
        sound = AudioSegment.from_mp3(audio_file_name)
        audio_file_name = audio_file_name.split('.')[0] + '.wav'
        sound.export(audio_file_name, format="wav")

def frame_rate_channel(audio_file_name):
    print(audio_file_name)
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        print(frame_rate, channels)
        return frame_rate,channels

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name, timeout=None)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

def google_transcribe(audio_file_name):
    file_name = filepath + audio_file_name
    mp3_to_wav(file_name)

    frame_rate, channels = frame_rate_channel(file_name)
    if channels > 1:
        stereo_to_mono(file_name)
        
    bucket_name = bucketname 
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name

    upload_blob(bucket_name, source_file_name, destination_blob_name)

    gcs_uri = 'gs://machine_feed_bucket' + "/" + audio_file_name
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    metadata = speech.RecognitionMetadata()
    metadata.interaction_type = speech.RecognitionMetadata.InteractionType.DISCUSSION
    metadata.microphone_distance = (
        speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD
    )
    metadata.recording_device_type = (
            speech.RecognitionMetadata.RecordingDeviceType.OTHER_OUTDOOR_DEVICE
    )

    config = speech.RecognitionConfig(
        encoding= speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        alternative_language_codes=['sv-SE'],
        enable_automatic_punctuation=True,
        enable_word_time_offsets = True,
        enable_word_confidence=True,
        metadata=metadata,
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=10000)
    result = response.results

    delete_blob(bucket_name, destination_blob_name)
    
    return result

def word_data_config(word_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results):
    new_filename = 'words_' + transcript_filename
    column_names = ["word","confidence", "start_time","end_time","sentence","sentence_confidence","date", "time", "original_audio"]
    df = pd.DataFrame(columns = column_names)

    for r in google_results:
        sentence_transcript = r.alternatives[0].transcript
        sentence_confidence = r.alternatives[0].confidence
        words_info = r.alternatives[0].words
        for word_info in words_info:
            current = word_info.word
            un_punctuated = current.translate(str.maketrans('', '', string.punctuation))
            confidence = word_info.confidence
            start_time = word_info.start_time
            end_time = word_info.end_time
            df.loc[len(df.index)] = [un_punctuated.lower(), confidence, start_time, end_time, sentence_transcript, sentence_confidence, date_recorded, ctime, audio_file_name]
            
    csv_filepath = word_filepath + new_filename
    df.to_csv(csv_filepath, index=True)

    
def sentence_data_config(sentence_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results):
    new_filename = 'sentence_' + transcript_filename
    column_names = ["sentence", "word_count", "confidence","date", "time", "original_audio"]
    df = pd.DataFrame(columns = column_names)
    
    for r in google_results:
        sentence_transcript = r.alternatives[0].transcript
        word_count = word_counter(sentence_transcript)
        confidence = r.alternatives[0].confidence
        df.loc[len(df.index)] = [sentence_transcript, word_count, confidence, date_recorded, ctime, audio_file_name]
    
    csv_filepath = sentence_filepath + new_filename
    df.to_csv(csv_filepath, index=True)

    
def transcript_data_config(transcript_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results):
    new_filename = 'transcript_' + transcript_filename
    column_names = ["transcript", "word_count", "date", "time", "original_audio"]
    df = pd.DataFrame(columns = column_names)
    
    full_transcript = ''
    for r in google_results:
        full_transcript += r.alternatives[0].transcript + ' '

    word_count = word_counter(full_transcript)
    
    df.loc[len(df.index)] = [full_transcript, word_count, date_recorded, ctime, audio_file_name]
    
    csv_filepath = transcript_filepath + new_filename
    df.to_csv(csv_filepath, index=True)


if __name__ == "__main__":
    for audio_file_name in os.listdir(filepath):
        if ".wav" in audio_file_name:
            print(audio_file_name)
            transcript_filename = audio_file_name.split('.')[0] + '.csv'
            date_recorded, ctime = get_date(audio_file_name)
            print('Getting results...')
            google_results = google_transcribe(audio_file_name)
            print('Writing transcripts...')
            word_data_config(word_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results)
            sentence_data_config(sentence_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results)
            transcript_data_config(transcript_filepath, transcript_filename, audio_file_name, date_recorded, ctime, google_results)
            print('Transcripts complete!')
            print(' ')
