import tensorflow as tf


def preprocess_image(image_path) -> tf.Tensor:
    image_path = image_path.numpy().decode("utf-8")

    # image_tensor = shape(110, 110, 1)
    image_tensor = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=1)
    image_tensor = tf.image.resize(image_tensor, [110, 110])
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0

    return image_tensor


def preprocess_class_label(class_label: int) -> tf.Tensor:

    # class_label = shape(n, 152)
    class_label = tf.convert_to_tensor(class_label - 1, dtype=tf.int32)
    one_hot_label = tf.one_hot(class_label, depth=152, dtype=tf.float32)

    return one_hot_label


def preprocess_data(image_path: str, class_label: int) -> tuple:
    try:
        image_tensor = preprocess_image(image_path)
        class_label = preprocess_class_label(class_label)
        return image_tensor, class_label

    except Exception as e:
        tf.print(f"Error processing {image_path}, {class_label}: {e}")
        return [
            tf.zeros((110, 110, 1), dtype=tf.float32),
            tf.zeros((152), dtype=tf.float32),
        ]


def preprocess_wrapper(image_path: str, class_label: int) -> tuple:
    image_tensor, class_label = tf.py_function(
        preprocess_data, [image_path, class_label], [tf.float32, tf.float32]
    )
    image_tensor.set_shape((110, 110, 1))
    class_label.set_shape((152))
    return image_tensor, class_label


def create_dataset(
    datas: list[tuple],
    batch_size: int = 128,
):
    image_paths = [str(t[1]) for t in datas]
    class_labels = [int(t[0]) for t in datas]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_labels))
    dataset = dataset.map(preprocess_wrapper)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
