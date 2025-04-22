python3 extract_textures.py --dataset ~/Pictures/bob_esponja_dataset/ --output ./output --grayscale false --filters_config src/filters.yaml --resize_size 512 --window_size 32 --split_factor 0.4

python3 segmentation.py --features_dir ./output --output ./output_segmented --limiar 0.2
