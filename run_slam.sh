#"abandonedfactory/Easy/P001"

# replica
for dataset in "room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4"
do

    DATASET_PATH="dataset/replica/Replica/"
    # dataset="office0"
    config="replica"
    OUTPUT_PATH="experiments/results/replica/$dataset"
    keyframe_th=0.7
    knn_maxd=99999.0
    overlapped_th=5e-4
    overlapped_th2=5e-5
    max_correspondence_distance=0.02
    trackable_opacity_th=0.05
    overlapped_th2=5e-5
    downsample_rate=10

    mkdir -p $OUTPUT_PATH/pcd

    python -W ignore gs_icp_slam.py --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --save_results
done


for dataset in  "soulcity/Easy/P001"
do
    DATASET_PATH="dataset/tartanair/scenes"
    config="tartanair"
    OUTPUT_PATH="experiments/results/$config/$dataset"
    keyframe_th=0.7
    knn_maxd=99999.0
    overlapped_th=5e-4
    overlapped_th2=5e-5
    max_correspondence_distance=0.02
    trackable_opacity_th=0.05
    overlapped_th2=1e-3
    downsample_rate=5

    echo "save results to $OUTPUT_PATH"
    mkdir -p $OUTPUT_PATH/pcd

    python -W ignore gs_icp_slam.py --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --save_results
done