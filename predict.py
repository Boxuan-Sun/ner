import os
import re
import pickle
import tensorflow as tf
from utils import create_model, get_logger, build_and_saved_model
from bilstm_crf import Model
from loader import input_from_line
from trainVersion import FLAGS, load_config

def exportModel(_):
    config = load_config(FLAGS.config_file)
    config["train"] = False
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    build_and_saved_model(Model, config, FLAGS.ckpt_path+"/ner.ckpt-95450", "/home/work/wangjing33/tensorflow_serving/models/ner/2")
    
def evaluate_line(_):
    config = load_config(FLAGS.config_file)
    config["train"] = False
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        while True:
            line = input("input sentence, please:")
            try:
                result = model.evaluate_line(sess, input_from_line(''.join(line.split(" ")), FLAGS.max_seq_len, tag_to_id), id_to_tag)
                print(result['entities'])
            except:
                print("please try again")
              
def evaluate_file(_):
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        cnt = 0
        print(FLAGS.ckpt_path)
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        with open("../Information-Extraction-Chinese-master/NER_IDCNN_CRF/data/example.demo", "r") as f, open("data/demo_result_1", "w") as fw:
            for line in f:
                line = line.strip().split("\t")
                if (len(line) > 2):
                    continue
                #print(line[0])
                text_list = re.split(r'[:：,，;；\"\'“”‘’、@# ]\s*', line[0])
                #print(text_list)
                entity = {}
                for t in text_list:
                    result = model.evaluate_line(sess, input_from_line(t, FLAGS.max_seq_len, tag_to_id), id_to_tag)
                    try:
                        json_result = result
                        if "entities" in json_result:
                            if len(json_result["entities"]) == 0:
                                continue
                            for item in json_result["entities"]:
                                word = item["word"]
                                type = item["type"]
                                if type == "PER":
                                    type = "celebrities"
                                if type == "ANL":
                                    type = "animals"
                                if type == "PLT":
                                    type = "plants"
                                if type == "DIS":
                                    type = "district"
                                if word not in entity:
                                    entity[word] = type
                                else:
                                    continue
                    except ValueError:
                        print("failed the JSON load")
                fw.write(line[1] + "\t" + line[0])
                if len(entity) != 0:
                    for k, v in entity.items():
                        fw.write("\t" + k + "\t" + v)
                else:
                    cnt += 1
                fw.write("\n")
            print("null_sentence_num:" + str(cnt))
            
def evaluate_file_fan(_):
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        cnt = 0
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        with open("/home/work/fanxinwen/ner/NER_DEMO.FAN/test831", "r") as f, open("data/demo_test_831", "w") as fw:
            for line in f:
                line = line.strip()
                entity = {}
                t = ''.join(line.split(" "))
                result = model.evaluate_line(sess, input_from_line(t, FLAGS.max_seq_len, tag_to_id), id_to_tag)
                try:
                    json_result = result
                    print(json_result["entities"])
                    if "entities" in json_result:
                        if len(json_result["entities"]) == 0:
                            continue
                        for item in json_result["entities"]:
                            word = item["word"]
                            type = item["type"]
                            if type == "PER":
                                type = "celebrities"
                            if type == "ANL":
                                type = "animals"
                            if type == "PLT":
                                type = "plants"
                            if type == "DIS":
                                type = "district"
                            if type == "POI":
                                type = "poi"
                            if word not in entity:
                                entity[word] = type
                            else:
                                continue
                except:
                    print("failed the JSON load")
                fw.write(t)
                if len(entity) != 0:
                    for k, v in entity.items():
                        fw.write("\t" + k + "\t" + v)
                else:
                    cnt += 1
                fw.write("\n")
            print("null_sentence_num:" + str(cnt))
           
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #tf.app.run(evaluate_file_fan)
    #tf.app.run(evaluate_file)
    #tf.app.run(exportModel)
    tf.app.run(evaluate_line)
                
                
