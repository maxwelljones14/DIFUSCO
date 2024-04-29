python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name "mst_diffusion_graph_categorical_mst50_final" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "" \
  --training_split "mst_small50-50.txt" \
  --validation_split "mst_small50-50.txt" \
  --test_split "mst_small50-50.txt" \
  --batch_size 32 \
  --num_epochs 1 \
  --ckpt_path "/home/maxwelljones14/DIFUSCO/models/mst_diffusion_graph_categorical_mst20/z46v4xqy/checkpoints/epoch=8-step=2304.ckpt" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --resume_weight_only \
  --save_numpy_heatmap \
  --mst_only \
  --save_gif --device 1

# for djikstra
#   --dijkstra_only \
#   --sparse_factor 6 \
#   --ckpt_path "/home/maxwelljones14/DIFUSCO/models/djikstra_diffusion_graph_categorical_mst50/sib06716/checkpoints/epoch=15-step=4096.ckpt"
# "/home/maxwelljones14/DIFUSCO/models/mst_diffusion_graph_categorical_mst20/4jukzkkw/checkpoints/epoch=6-step=1792.ckpt" \
# for mst
# "/home/maxwelljones14/DIFUSCO/models/mst_diffusion_graph_categorical_mst20/z46v4xqy/checkpoints/epoch=8-step=2304.ckpt" \

# python -u difusco/train.py \
#   --task "tsp" \
#   --wandb_logger_name "djikstra_diffusion_graph_categorical_mst50" \
#   --diffusion_type "categorical" \
#   --do_train \
#   --learning_rate 0.0002 \
#   --weight_decay 0.0001 \
#   --lr_scheduler "cosine-decay" \
#   --storage_path "" \
#   --training_split "dijkstra50-50.txt" \
#   --validation_split "dijkstra_small50-50.txt" \
#   --test_split "dijkstra_small50-50.txt" \
#   --batch_size 64 \
#   --num_epochs 20 \
#   --validation_examples 8 \
#   --inference_schedule "cosine" \
#   --inference_diffusion_steps 200 \
#   --do_test \
#   --dijkstra_only \
#   --sparse_factor 6 \
#   --device 0


# mst training: 10 epochs cosine-decay mst50-50.txt mst_small50-50.txt --mst_only
