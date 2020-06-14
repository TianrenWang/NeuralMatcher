JOB_NAME=jobname
STAGING_BUCKET=gs://neuralsearch
REGION=us-central1
DATA_DIR=gs://neuralsearch/raw_data
DATA_NAME=raw_data
OUTPUT_PATH=gs://neuralsearch/model
ENCODED_DATA=gs://neuralsearch/encoded_data
PACKAGE=gs://neuralsearch/tensorflow-datasets-3.1.0.tar.gz,gs://neuralsearch/nltk-3.5.zip

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $STAGING_BUCKET \
    --runtime-version 1.15 \
    --python-version 3.7 \
    --scale-tier BASIC_TPU \
    --module-name NeuralSearchEngine.estimator \
    --package-path NeuralSearchEngine/ \
    --region $REGION \
    --packages $PACKAGE \
    -- \
    --data_dir=$DATA_DIR \
    --data_name=$DATA_NAME \
    --model_dir=$OUTPUT_PATH \
    --encoded_data_dir=$ENCODED_DATA \
    --train_steps=10000 \
    --vocab_level=15 \
    --dropout=0.1 \
    --heads=8 \
    --abstract_len=512 \
    --title_len=60 \
    --batch_size=32 \
    --layers=4 \
    --depth=512 \
    --feedforward=512 \
    --train=True \
    --predict=True \
    --predict_samples=8800 \
    --description="Put the experimental description here" \

  $@