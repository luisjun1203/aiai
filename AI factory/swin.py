import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 이 부분이 튜플로 정의되어 있다고 가정합니다.
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

        def call(self, x, mask=None):
        # q, k, v 계산
            B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            qkv = tf.split(self.qkv(x), 3, axis=-1)
            q, k, v = [tf.reshape(tensor, [B_, N, self.num_heads, C // self.num_heads]) for tensor in qkv]
            q = q * self.scale
            k = tf.transpose(k, perm=[0, 2, 1, 3])  # 주의: 여기서는 k의 차원을 바꿔줍니다.

        # Attention 계산
            attn = tf.matmul(q, k)

        # Relative position bias 가져오기 및 적용
            relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
            relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])  # 수정: 차원 변경
            relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # 수정: 차원 변경
            relative_position_bias = tf.expand_dims(relative_position_bias, 0)
            attn += relative_position_bias

            if mask is not None:
                attn += mask

            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)

            # 출력 계산
            attn = tf.matmul(attn, v)
            attn = tf.transpose(attn, perm=[0, 2, 1, 3])
            attn = tf.reshape(attn, [B_, N, C])
            x = self.proj(attn)
            x = self.proj_drop(x)

            return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


# class SwinTransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#         self.prefix = prefix

#         self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
#         self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
#                                     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
#         self.drop_path = DropPath(
#             drop_path_prob if drop_path_prob > 0. else 0.)
#         self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        drop=drop, prefix=self.prefix)

#     def build(self, input_shape):
#         if self.shift_size > 0:
#             H, W = self.input_resolution
#             img_mask = np.zeros([1, H, W, 1])
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             img_mask = tf.convert_to_tensor(img_mask)
#             mask_windows = window_partition(img_mask, self.window_size)
#             mask_windows = tf.reshape(
#                 mask_windows, shape=[-1, self.window_size * self.window_size])
#             attn_mask = tf.expand_dims(
#                 mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
#             attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
#             attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
#             self.attn_mask = tf.Variable(
#                 initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
#         else:
#             self.attn_mask = None

#         self.built = True

#     def call(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.get_shape().as_list()
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = tf.reshape(x, shape=[-1, H, W, C])

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = tf.roll(
#                 x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)
#         x_windows = tf.reshape(
#             x_windows, shape=[-1, self.window_size * self.window_size, C])

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)

#         # merge windows
#         attn_windows = tf.reshape(
#             attn_windows, shape=[-1, self.window_size, self.window_size, C])
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = tf.roll(shifted_x, shift=[
#                         self.shift_size, self.shift_size], axis=[1, 2])
#         else:
#             x = shifted_x
#         x = tf.reshape(x, shape=[-1, H * W, C])

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_prob = drop_path_prob
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=prefix)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop, prefix=prefix)

    def call(self, x):
        # x: [batch_size, num_patches, dim]
        shortcut = x
        x = self.norm1(x)
        # Apply window-based self attention
        x = self.attn(x)

        # Apply drop path
        x = shortcut + self.drop_path(x)
        shortcut = x

        # Apply MLP
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)

        return x

# input_tensor = tf.random.normal((1, 4096, 96, 1))  # 입력 텐서의 모양을 (1, 4096, 96, 1)로 변경

input_tensor = tf.random.normal((1, 256, 256, 576))  # 입력 텐서의 모양을 (1, 4096, 96, 96)로 변경

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        # patch_merging 레이어에 입력되는 텐서의 형태 조정
        print("Input tensor shape:", x.shape)  # 입력 텐서의 모양 출력
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Height and width must be even. Received: H={H}, W={W}"
        assert C % 4 == 0, f"The number of channels must be divisible by 4. Received: {C}"
        
        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4C

        # 패치 병합 후 차원 감소 및 정규화
        x = self.norm(x)
        x = self.reduction(x)

        return x

model = PatchMerging(dim=96)
output_tensor = model(input_tensor)
# print("Output tensor shape:", output_tensor.shape)

class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = tf.keras.Sequential()

        for i in range(depth):
            self.blocks.add(SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path_prob=drop_path_prob,
                norm_layer=norm_layer,
                prefix=f'{prefix}/block_{i}')
            )

        if downsample is not None:
            self.downsample = downsample(
                dim=dim * 2,
                norm_layer=norm_layer,
                prefix=f'{prefix}/downsample_{i}')

    def call(self, x):
        x = self.blocks(x)  # Sequential 모델이므로, 단순히 x를 blocks에 통과시킵니다.

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(256, 256), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        # img_size는 튜플 (높이, 너비) 형태로 받음
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x
    
import tensorflow as tf
from keras import layers, models

class SwinTransformerWithUpsample(tf.keras.Model):
    def __init__(self, img_size=(256, 256), patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        self.pos_drop = layers.Dropout(rate=0.0)

        self.transformer_layers = []
        for i, layer_depth in enumerate(depths):
            layer_dim = embed_dim * (2 ** i)  
            layer = BasicLayer(dim=layer_dim,
                               input_resolution=(img_size[0] // (2 ** i), img_size[1] // (2 ** i)),
                               depth=layer_depth,
                               num_heads=num_heads[i],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=True, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path_prob=0.1,
                               norm_layer=layers.LayerNormalization,
                               downsample=PatchMerging if i < len(depths) - 1 else None,
                               prefix=f'layer_{i}')
            self.transformer_layers.append(layer)
            
        self.norm = layers.LayerNormalization(epsilon=1e-5, name='norm')
        
        # Upsample 및 최종 레이어 정의
        self.upsample_layers = [
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(embed_dim // 2, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(1, 3, padding='same'),
        ]
        self.final_conv = layers.Conv2D(1, 1, padding='same')  # 최종 Conv 레이어
        self.sigmoid = layers.Activation('sigmoid')  # Sigmoid 활성화 함수

    def call(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)
        
        for layer in self.upsample_layers:  # Upsample 레이어 적용
            x = layer(x)
        
        x = self.final_conv(x)  # 최종 Conv 레이어 적용
        x = self.sigmoid(x)  # Sigmoid 활성화 함수 적용
        
        return x

model = SwinTransformerWithUpsample(img_size=(256, 256), patch_size=4, in_chans=3, embed_dim=96)
input_tensor = tf.random.normal((1, 256, 256, 576))  # 입력 텐서의 모양을 (1, 4096, 96, 96)로 변경
model(input_tensor)
model.summary()
