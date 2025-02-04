import pyaudio
import wave
import torch
import torchaudio
import soundfile as sf




def record():
    chunk = 1024 
    sample_format = pyaudio.paInt16  
    channels = 2
    fs = 44100  
    seconds = 5
    filename = "output.wav"

    p = pyaudio.PyAudio() 

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe(filename):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model=bundle.get_model()
    label=bundle.get_labels()

    filename=filename
    data,sample_rate=sf.read(filename,dtype='float32')
    waveform=torch.from_numpy(data)

    if waveform.ndim>1:
        waveform=waveform.mean(dim=1)

    if sample_rate != bundle.sample_rate:
        waveform=torchaudio.functional.resample(waveform,sample_rate,bundle.sample_rate)
    
    waveform= waveform.unsqueeze(0)

    with torch.inference_mode():
        emissions, _ = model(waveform)
    emissions=torch.softmax(emissions[0],dim=1)
    tokens=torch.argmax(emissions,dim=-1)
    transcription = []
    prev_token = None
    blank_token = label.index('|') 

    for token in tokens:
        if token != prev_token and token != blank_token:
            transcription.append(label[token])
        prev_token = token

    transcribed_text = ''.join(transcription)
    print("Transcription:", transcribed_text)

    return transcribed_text


def main():
    #filename=record()
    transcription=transcribe('output.wav')
    print(transcription)
if __name__=="__main__":
    main()