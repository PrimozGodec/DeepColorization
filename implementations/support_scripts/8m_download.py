import csv
import json
import os
from random import shuffle
import pickle


""" all nature video extractor"""

save_in = "../../processed_data/nature_videos_links.pkl"

if not os.path.isfile(save_in):

    nature_label = 22  # from file with label names

    with open("../../data/train_labels.csv", "r") as f:
        reader = csv.reader(f)
        all_videos = list(reader)

    # print(all_videos[0])
    # split labels
    for i, video in enumerate(all_videos):
        all_videos[i][1] = video[1].split()

    # get all nature videos
    nature_videos = [x[0] for x in all_videos if str(nature_label) in x[1]]
    shuffle(nature_videos)

    with open(save_in, "wb") as f:
        pickle.dump(nature_videos, f)

with open(save_in, "rb") as f:
    nature_videos = pickle.load(f)


""" download videos """
# extract videos length
save_in = "../../processed_data/nature_videos_information.pkl"

if os.path.isfile(save_in):
    with open(save_in, "rb") as f:
        video_info = pickle.load(f)
else:
    video_info = dict()

total_length = sum(x["duration"] for _, x in video_info.items())

import youtube_dl

info = {}
ydl_opts = {}
ydl = youtube_dl.YoutubeDL({'outtmpl': '../../../videos_dataset/%(id)s.%(ext)s'})
with open(save_in, "wb") as f:
    with ydl:
        for video in nature_videos:
            if video not in video_info:
                try:
                    result = ydl.extract_info(
                        'https://www.youtube.com/watch?v=' + video,
                        download=False # We just want to extract the info
                    )
                except youtube_dl.utils.DownloadError:
                    continue  # continue if do not exist

                ydl.download(['https://www.youtube.com/watch?v=' + video])

                print(video_info)
                # save info about allready downloaded
                video_info[video] = {"duration": result["duration"]}
                pickle.dump(video_info, f)

                total_length += result["duration"]
                if total_length > 4000:
                    exit()







