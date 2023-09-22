This repository contains the code for the paper:

"Covid vaccine is against Covid but Oxford vaccine is made at Oxford!" Semantic Interpretation of Proper Noun Compounds\
Keshav Kolluru, Gabriel Stanovsky, Mausam\
EMNLP 2022

# Installation

Use the following commands to install the requirements required for running the code:

```
pip install -r requirements.txt
cd transformers
pip install --editable .
cd ..
```

# Data

The data directory contains all the required files collected and used for the project.


# Model Training

```
model_type= uniGen (or) clsGen
data= rand (or) nns
knowledge= base (or) sentence (or) knowledge_nnp (or) knowledge_nn (or) ner
seed=42

python run.py --model_name_or_path t5-base --output_dir models/${model_type}_${data}/${knowledge}/seed${seed} --overwrite_output_dir --do_predict --predict_with_generate --overwrite_cache --metric_type sacrebleu --num_train_epochs 10  --per_device_train_batch_size 16 --data_type ${regime}_${knowledge} --model_type ${model_type} --seed ${seed}
```

# Open IE Integration

```
bash openie_nci.sh
```
