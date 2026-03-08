
K_VALUES=(10)

for k in "${K_VALUES[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python /root/Desktop/workspace/kml/1_User_encoder/test/test_baseline_e5v.py \
        --test_file /root/Desktop/workspace/kml/1_User_encoder/dataset/user_split/media/test.json \
        --image_embedding_dir /root/Desktop/workspace/kml/MovieLens/image_embeddings/1031_Media_5000_image_embeddings_flat \
        --model_name royokong/e5-v \
        --output_dir /root/Desktop/workspace/kml/1_User_encoder/result_new/baseline_e5v_1111/user_split_kfold_k${k} \
        --image_source_dir /root/Desktop/workspace/kml/MovieLens/images/1031_Media_5000_images_flat \
        --image_output_dir /root/Desktop/workspace/kml/1_User_encoder/image_result_new_1111/baseline_e5v/user_split_kfold_k${k}  \
        --top_k 50 \
        --batch_size 128 \
        --seed 42 \
        --target_policy filter_last \
        --allowed_subs "favorite media;favourite media;favorite actors and directors;favourite actors and directors" \
        --skip_on_mismatch \
        --eval_gt_similarity
done