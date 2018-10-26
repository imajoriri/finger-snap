
import wave
import pyaudio

# チャンク数を指定
def play_wave(file_name):

    buffer_size = 4096
    wav_file = wave.open ( file_name , 'rb' )
    p = pyaudio.PyAudio ()
    stream = p.open (
                     format = p.get_format_from_width ( wav_file . getsampwidth ()) ,
                     channels = wav_file.getnchannels () ,
                     rate = wav_file.getframerate () ,
                     output = True
                     )
    remain = wav_file.getnframes ()
    while remain > 0:
        buf = wav_file.readframes ( min ( buffer_size , remain ))
        stream.write ( buf )
        remain -= buffer_size
    
    stream.close ()
    p.terminate ()
    wav_file.close ()
