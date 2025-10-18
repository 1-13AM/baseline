# scripts to extract frames from videos

python -m utils.extract_frames --input_folder INPUT_FOLDER \
                               --output_base_folder OUTPUT_BASE_FOLDER

python -m utils.check_num_frames --base_path OUTPUT_BASE_FOLDER \
                                 --n 25