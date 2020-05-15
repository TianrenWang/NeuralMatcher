JOB_NAME=neuralmatcher13
STAGING_BUCKET=gs://neuralsearch
REGION=us-central1
DATA_DIR=gs://neuralsearch/raw_data
OUTPUT_PATH=gs://neuralsearch/model
ENCODED_DATA=gs://neuralsearch/encoded_data
PACKAGE=gs://neuralsearch/tensorflow-datasets-3.1.0.tar.gz

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
    --model_dir=$OUTPUT_PATH \
    --encoded_data_dir=$ENCODED_DATA \
    --train_steps=1 \
    --vocab_level=15 \
    --dropout=0.1 \
    --heads=8 \
    --abstract_len=512 \
    --title_len=60 \
    --batch_size=1 \
    --layers=4 \
    --depth=256 \
    --feedforward=512 \
    --train=True \
    --predict=True \
    --predict_samples=10 \
    --description="Put the experimental description here" \

  $@