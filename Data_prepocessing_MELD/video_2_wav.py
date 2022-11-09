from moviepy.editor import *

rootdir = '/home/shixiaohan-toda/Documents/DataBase/journal_Data'
Data_dir = rootdir + '/MELD.Raw/'

def trans_all_file(file_path,train_video):
    files_path = file_path + train_video
    for filepath in os.listdir(files_path):
        print(filepath)
        if(filepath[0] != '.'):
            modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
            datapath = os.path.join(modpath, files_path + filepath)
            video = VideoFileClip(datapath)
            audio = video.audio
            data_name = filepath[:-4] + '.wav'
            wav_path = modpath + '/dev_wav/'+ data_name
            print(audio.duration)
            audio.write_audiofile(wav_path)

train_video = 'dev/dev_splits_complete/'
trans_all_file(Data_dir,train_video)
