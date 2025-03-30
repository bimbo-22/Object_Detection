import os 
import yaml 
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import config_util

params = yaml.safe_load(open('params.yaml'))['preprocess']

def create_tf_example(image_path, label_path, label_map_dict):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    filename = os.path.basename(image_path).encode('utf8')
    image_format = b'jpg' if image_path.endswith('jpg') else b'png'
    
    with open(label_path, 'r') as file:
        lines = file.readlines()
        
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    height, width = 640, 640
    for line in lines:
        clas, x_center, y_center, w, h = map(float, line.strip().split())
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x_min = int(x_center - w / 2)
        x_max = int(x_center + w / 2)
        y_min = int(y_center - h / 2)
        y_max = int(y_center + h / 2)
        xmins.append(x_min / width)
        xmaxs.append(x_max / width)  
        ymins.append(y_min / height)
        ymaxs.append(y_max / height)
        classes_text.append(label_map_dict[int(clas)].encode('utf8'))
        classes.append(int(clas)+1)
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def preprocess(input_images, input_labels, output_tfrecord):
    label_map_dict = label_map_util.get_label_map_dict("label_map.pbtxt")
    writer = tf.io.TFRecordWriter(output_tfrecord)
    
    for filename in os.listdir(input_images):
        if filename.endswith((",jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_images, filename)
            label_path = os.path.join(input_labels, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
            if os.path.exists(label_path):
                tf_example = create_tf_example(image_path, label_path, label_map_dict)
                writer.write(tf_example.SerializeToString())
                
    writer.close()
    
if __name__ == "__main__":
    preprocess(params['input_images'], params['input_labels'], params['output_tfrecord'])