import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, LayerNormalization
from keras.preprocessing import Resizing
from keras.models import Model

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = self.add_weight("pos_embedding", shape=[1, num_patches + 1, projection_dim])

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding[:, :self.num_patches, :]
        return encoded

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=8, transformer_units=[2048, 1024], dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.dense_proj = tf.keras.Sequential([Dense(units, activation=tf.nn.gelu) for units in transformer_units])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training):
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm2(proj_input + proj_output)

class VisionTransformer(Model):
    def __init__(self, image_size=72, patch_size=6, num_layers=12, num_classes=10, projection_dim=64):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(num_patches=num_patches, projection_dim=projection_dim)
        self.transformer_blocks = [TransformerBlock(projection_dim) for _ in range(num_layers)]
        self.flatten = Flatten()
        self.dropout = Dropout(0.1)
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = Resizing(inputs, size=(self.patch_size, self.patch_size))
        embeddings = self.patch_embedding(patches)
        transformed = embeddings
        for block in self.transformer_blocks:
            transformed = block(transformed)
        representation = self.flatten(transformed)
        representation = self.dropout(representation)
        logits = self.classifier(representation)
        return logits