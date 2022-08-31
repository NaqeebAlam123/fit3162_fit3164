# Adapted from https://gist.github.com/arjunsharma97/0ecac61da2937ec52baf61af1aa1b759

import os
from pydub import AudioSegment

AUDIO_FOLDER = '../../data/audio/M030 copy'

def main():
    os.remove(f'{AUDIO_FOLDER}/.DS_Store')
    formats_to_convert = ('.m4a',)
    for sub_dir in os.listdir(AUDIO_FOLDER):
        for org_audio_filename in os.listdir(os.path.join(AUDIO_FOLDER, sub_dir)):
            _path = os.path.join(AUDIO_FOLDER, sub_dir, org_audio_filename)
            if not os.path.isfile(_path):
                print(f'{org_audio_filename} is not a file')
                continue
            if org_audio_filename.endswith(formats_to_convert):
                _, file_extension = os.path.splitext(_path)
                file_extension = file_extension.replace('.', '')
                try:
                    track = AudioSegment.from_file(_path,
                            file_extension)
                    wav_filename = org_audio_filename.replace(file_extension, 'wav')
                    wav_path = os.path.join(AUDIO_FOLDER, sub_dir, wav_filename)
                    print('CONVERTING: ' + str(_path))

                    track.export(wav_path, format='wav')
                    os.remove(_path)
                except:
                    print("ERROR CONVERTING " + str(_path))
    return


if __name__ == '__main__':
    main()
