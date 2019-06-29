# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add( node1, node2 )

sess = tf.compat.v1.Session()

print("sess.run( node1 + node2 ): ", sess.run( node3 ) )