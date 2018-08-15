
import tensorflow as tf

# based on https://arxiv.org/pdf/1807.03247.pdf
# adds x, y coordinates normalized to [-1, 1] as additional 
# features for a given 2D input
class CoordConvTransform2D(tf.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        _, w, h, _ = inputs.get_shape()
        bs = tf.shape(inputs)[0]
        
        # Get indices
        indices = tf.where(tf.ones(tf.stack([bs, w, h])))
        indices = tf.cast(indices, tf.float32)
        canvas = tf.reshape(indices, tf.stack([bs, w, h, 3]))[..., 1:]
        # Normalize the canvas
        w_max = w
        h_max = h
        if w > 1:
            w_max = w - 1
        if h > 1:
            h_max = h - 1
        canvas = canvas / tf.cast(tf.reshape(tf.stack([w_max, h_max]), [1, 1, 1, 2]), tf.float32)
        canvas = (canvas * 2) - 1
        
        # Concatenate channel-wise
        outputs = tf.concat([inputs, canvas], axis=-1)
        return outputs

def coord_conv_transform2d(inputs):
    layer = CoordConvTransform2D()
    outputs = layer.apply(inputs)
    return outputs