## Data Preparation

### Tea Dataset
Prepare tea dataset directory with the following structure:

```
your_data_path/
|–– tea/
|   |–– data/
|   |   |–– images/
|   |   |–– train.json
|   |   |–– val.json
|   |   |–– test.json
|   |   |-- reformat_train.json
|   |   |-- reformat_val.json
|   |   |-- reformat_test.json
```



gt
|–– clevr_total_change_captions_reformat.json
|–– spot_total_change_captions_reformat.json
```


# Pretrained Weight

```sh
cd ckpts
mkdir pretrained
mkdir trained
```

You can download the [Pretrained Weights](https://drive.google.com/drive/folders/1qOYVpZy57clJPF6AThsnO0Tfy4zq-gg1?usp=sharing) from the tea-based IDC Adaptation and the [Trained Weights](https://drive.google.com/drive/folders/18UfIvwKt0EE14EbogJycMmANpUJtsZbE?usp=sharing) from the tea-based IDC Finetuning. You would get
 

## How to Run 

>`--features_path` is the data root path
> 
> `--pretrained_clip_name` can be set with `ViT-B/32` 
> 
> `--resume_model` can be used to reload the saved optimizer state to continuely train the model, **Note**: need to set the corresponding chechpoint via `--init_model` simultaneously. 

Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```


## Adapation

Experiments are conducted on 2 **NVIDIA RTX A5000 GPUs**. Time required for each task is less than 24h.

### Tea Dataset

```sh
DATA_PATH=[Tea data path]
python -m torch.distributed.launch --nproc_per_node=2 main_task_retrieval.py \
--do_train \
--num_thread_reader=4 \    
--batch_size=128 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_tea_retrieval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 128 \
--datatype Tea \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 
```

>The Text-to-Image-Pair retrieval results are close to 
```
R@1: 26.8 - R@5: 58.7 - R@10: 70.0
```
>The Image-Pair-to-Text retrieval results are close to 
```
R@1: 46.4 - R@5: 83.0 - R@10: 86.6
```


## Finetuning

Time required for each task is less than 24h.

### Tea dataset 

Reproducing the results on 2 **NVIDIA RTX A5000**.

```sh
DATA_PATH=[tea dataset path]
python -m torch.distributed.launch --nproc_per_node=1 main_task_caption.py \
--do_train \
--num_thread_reader=4 \  # Please don't change this value when reproducing the results
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_tea_caption \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 64 \
--datatype clevr \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/pretrained/pytorch_model.bin.clevr \
--seed 2021 
```

>The best results are obtained at epoch 19 
```
BLEU_1: 0.8648, BLEU_2: 0.7797, BLEU_3: 0.6758, BLEU_4: 0.5687
METEOR: 0.3840, ROUGE_L: 0.7643, CIDEr: 1.5075
```

Reproducing the results on Two **NVIDIA V100.

```sh
DATA_PATH=[tea dataset path]
python -m torch.distributed.launch --nproc_per_node=2 main_task_caption.py \
--do_train \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=64 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_tea_caption_multigpu \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 64 \
--datatype clevr \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/pretrained/pytorch_model.bin.clevr \
--seed 2021 
```

>The best results are obtained at Epoch 26
```

BLEU_1: 0.8573, BLEU_2: 0.7761, BLEU_3: 0.6734, BLEU_4: 0.5663
METEOR: 0.3900, ROUGE_L: 0.7640, CIDEr: 1.5039

```
The latest results values for tea based retrieval system will be published soon

Due to NDA signed for the company results wont be to published.

get the latest trained weights on our shared drive.

added the results on drive



