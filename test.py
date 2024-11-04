import engine.Paramenters as Paramenters
import os
import json
import cv2

import engine.MT as MT
def Generate_videos_x(args, save_path):
    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    Pix_MRs = [mr for mr in MRs if mr["type"] == "Pix2Pix"]
    diffusion_MRs = [mr for mr in MRs if mr["type"] == "diffusion"]
    length = len(Pix_MRs) + len(diffusion_MRs)
    save_dir, save_dir_1, save_dir_2 = MT.get_results_dir(args)
    test_save_path = os.path.join(save_dir_2, save_path)
    with open(os.path.join(args.data_file, "results", "info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    args.pre_series = 25

    # Define the gap between images (in pixels)
    gap = 20

    for i in range(length):
        image_dir = os.path.join(save_dir_2, save_path, str(i), "images")
        video_dir = os.path.join(save_dir_2, save_path, str(i), "videos")
        Exp_2_save = os.path.join(save_dir_2, save_path, str(i), "Exp_p")
        cont = 0
        for idx in range(0, len(data_lists), args.pre_series):
            start_index = idx
            end_index = idx + args.pre_series
            for idxx in range(start_index, end_index):
                original_frame_path = os.path.join(save_dir, os.path.basename(data_lists[idxx]['Image File']))
                generated_frame_path = os.path.join(video_dir, os.path.basename(data_lists[idxx]['Image File']))

                original_frame = cv2.imread(original_frame_path)
                generated_frame = cv2.imread(generated_frame_path)

                original_frame_resized = cv2.resize(original_frame, (640, 320), interpolation=cv2.INTER_AREA)
                generated_frame_resized = cv2.resize(generated_frame, (640, 320), interpolation=cv2.INTER_AREA)

                # Create vertical comparison frame with gap
                comparison_frame = np.vstack((
                    original_frame_resized,
                    np.full((gap, 640, 3), 255, dtype=np.uint8),  # White gap
                    generated_frame_resized
                ))
                frames.append(comparison_frame)

            video_path = os.path.join(Exp_2_save, f"{cont}.mp4")
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in frames:
                video.write(frame)
            video.release()
            cont += 1


if __name__ == "__main__":
    args = Paramenters.parse_args()
    #MT.find_test_images_1(args)
    with open(os.path.join("Data", "Texas_final.json"), 'r', encoding='utf-8') as file:
        MRs = json.load(file)
    print(1)

    ""