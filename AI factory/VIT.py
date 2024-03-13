from keras.layers import Input, Conv2D, Reshape, Dense, Flatten
from keras.models import Model
import tensorflow as tf

def create_vit_segmenter(input_shape=(256, 256, 3), num_patches=256, projection_dim=64, num_heads=4, transformer_layers=4, num_classes=1):
    inputs = Input(shape=input_shape)
    # 이미지를 패치로 분할
    patches = Reshape((num_patches, input_shape[2] * (input_shape[0] // int(num_patches**0.5)) * (input_shape[1] // int(num_patches**0.5))))(inputs)
    # 패치 임베딩
    patch_embeddings = Dense(units=projection_dim)(patches)

    # 위치 임베딩 추가
    position_embeddings = tf.Variable(tf.zeros(shape=(1, num_patches, projection_dim)))
    embeddings = patch_embeddings + position_embeddings
    
    # Transformer 레이어
    for _ in range(transformer_layers):
        # Multi-head Self-Attention (MSA) / LayerNorm (LN) / MLP
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(embeddings, embeddings)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(embeddings + attention_output)
        x = tf.keras.layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x)
        embeddings = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + embeddings)
    
    # 패치 기반 분할 마스크 생성
    patch_outputs = Dense(units=(input_shape[0] * input_shape[1] // num_patches) * num_classes, activation='sigmoid')(embeddings)
    # 분할 마스크로 재구성
    segmentation_mask = Reshape((input_shape[0], input_shape[1], num_classes))(patch_outputs)
    
    model = Model(inputs=inputs, outputs=segmentation_mask)
    return model

# 모델 생성 및 컴파일
vit_segmenter = create_vit_segmenter()
vit_segmenter.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
vit_segmenter.summary()