export BERT_CONFIG_PATH="/home/shan/data/wiki_for_bert/bert_config.json"
#export INPUT_FILE="/home/shan/data/wiki_for_bert/tfrecord/part-0000*"
#export CHECKPOINT_PATH="/home/shan/data/wiki_for_bert/tf1_ckpt"
export INPUT_FILE="hdfs:///user/yushan/wiki_for_bert/tfrecord/part*"
# export CHECKPOINT_PATH="hdfs:///user/yushan/wiki_for_bert/tf1_ckpt"
export CHECKPOINT_PATH="ckpt/tf1_ckpt"
# export PYSPARK_DRIVER_PYTHON=~/anaconda3/envs/tf1-py36/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server:
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
export HADOOP_HDFS_HOME=$HADOOP_HOME
# export HADOOP_HDFS_HOME=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/


TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
nohup python run_pretraining.py \
  --bert_config_file=$BERT_CONFIG_PATH \
  --output_dir=/tmp/output/ \
  --input_file=$INPUT_FILE \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=$CHECKPOINT_PATH/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=10 \
  --num_warmup_steps=5 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --train_batch_size=160 \
  --max_eval_steps=5 \
  --cluster_mode=yarn \
  --num_executors=10 > logs/yarn-10-executors-batch-size-160-2.log 2>&1 &
