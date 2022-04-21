import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util

read_features = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

tfrecord = tf.data.TFRecordDataset("/home/katerina/Documents/Masters-Thesis/annotations/val.record")
for record in tfrecord:
    #extract feature
    example = tf.train.Example()
    example.ParseFromString(record)
    print(example)

    tf.parse_single_example(serialized=example,
                                features=read_features)



    image = example.features.feature["encoded"].bytes_list.value[0]

    # save image to file
    print(image)

'''
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    image = example.features.feature["encoded"].bytes_list.value[0]

    # save image to file
    print(image)
'''
