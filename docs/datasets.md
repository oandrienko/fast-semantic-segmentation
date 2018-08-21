# Dataset Requirements

This project uses the [TFRecord format](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) to consume data in the training and evaluation process. Creating a TFRecord from raw image files is pretty straight forward and will be covered here.

*Note:* **This project includes a script for creating a TFRecord for Cityscapes**, but not other datasets. To create your own TFRecord script, use the one in this project as are reference. Please note that the fields in the record must be the same as defined in the script. Read below for details.

## Creating TFRecords for Cityscapes

In order to download the Cityscapes dataset, you must first register with their [website](https://www.cityscapes-dataset.com/). After this, make sure to download both `leftImg8bit` and `gtFine`. You should end up with a folder that will be in the structure

```
+ cityscapes
 + leftImg8bit
 + gtFine
```

Next, in order to generate training labels for the dataset, clone cityScapesScripts project

```
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
```

Then from the root of your cityscapes dataset run

```
# must have $CITYSCAPES_ROOT defined
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

Finally, you can now run the conversion script provided in this project with

```
python create_cityscapes_tfrecord.py \
	--cityscapes_dir $CITYSCAPES_ROOT \
	--output_dir /tmp \
	--split_type train # must be `train` or `val`
```

Change the `split_type` flag for switching between val and train sets. After running the script for both the train and val set, you should end up with tfrecord files named `cityscapes_train.record` and `cityscapes_val.record`.


## Writing your own TFRecord conversion script

Creating a TFRecord for your own dataset is a bit more involved. This following will be a general procedure to help you write your own conversion script.

### Dataset Requirements

For every example in your dataset, you should have the following information:

* An RGB image for the dataset encoded as jpeg or png.
* A Grayscale groundtruth image encoded as jpeg or png.

### Creating a tf. Example

Each training pair in your semantic segmentation dataset will require a tf. Example proto defined within the record. For example, consider creating one for a single PNG image. This can be done with

```python
def create_tf_example(encoded_image, encoded_label, full_image_path):

 height = 1024
 width = 2048
 input_channels = 3

 Example = tf.train. Example(
 features=tf.train. Features(feature={
 'image/encoded': _bytes_feature(encoded_image),
 'image/filename': _bytes_feature(full_image_path.encode('utf8')),
 'image/height': _int64_feature(height),
 'image/width': _int64_feature(width),
 'image/channels': _int64_feature(3),
 'image/segmentation/class/encoded': _bytes_feature(encoded_label),
 'image/format': _bytes_feature('png'.encode('utf8')),
 'image/segmentation/class/format':_bytes_feature('png'.encode('utf8'))
 }))
 return example
```

The keys in the dictionary passed to `tf.train. Features` should match if you wish to use your record with this project.

In general, the conversion script will have the following structure

```python
import tensor flow as tf

flags = tf.app.flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

#
# Util functions
#
def _bytes_feature(values):
 return tf.train. Feature(
 bytes_list=tf.train. BytesList(value=[values]))

Def _int64_feature(values):
 if not is instance(values, (tuple, list)):
 values = [values]
 return tf.train. Feature(int64_list=tf.train. Int64List(value=values))

#
# Main function for creating an example
#
def create_tf_example(example):
 # TODO(user): Populate the following variables from your example.
 height = None # Image height
 width = None # Image width
 channels = None # Input image Channels
 filename = None # Filename of the image. Empty if image is not from file
 encoded_image = None # Encoded image bytes
 encoded_label = None # Encoded ground truth image bytes
 format = None # b'jpeg' or b'png'

 example = tf.train. Example(features=tf.train. Features(feature={
	 'image/encoded': _bytes_feature(encoded_image),
	 'image/filename': _bytes_feature(filename.encode('utf8')),
	 'image/height': _int64_feature(height),
	 'image/width': _int64_feature(width),
	 'image/channels': _int64_feature(channels),
	 'image/segmentation/class/encoded': _bytes_feature(encoded_label),
	 'image/format': _bytes_feature(format.encode('utf8')),
	 'image/segmentation/class/format':_bytes_feature(format.encode('utf8'))
 }))
 return example

def main(_):
 writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

 # TODO(user): Write code to read in your data set to examples variable

 For example in examples:
 example = create_tf_example(example)
 writer.write(example. SerializeToString())

 writer.close()

if __name__ == '__main__':
 tf.app.run()

```

Use `create_cityscapes_tfrecord.py` as a reference when you are writing your own converter.
