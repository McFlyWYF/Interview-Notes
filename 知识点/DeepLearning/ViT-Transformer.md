
### ViT模型


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
```

#### 超参数


```python
leanring_rate = 0.001
weight_decay = 0.001
batch_size = 256
num_epochs = 100
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) **2
projection_dim = 64
num_heads = 4

transformer_uints = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_uints = [2048, 1024]
```

#### 数据


```python
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
```

    x_train shape: (50000, 32, 32, 3)
    x_test shape: (10000, 32, 32, 3)
    

* 数据扩充


```python
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.Resizing(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(factor=0.02),
        layers.experimental.preprocessing.RandomZoom(
            height_factor = 0.2, width_factor = 0.2),
    ],
    name = 'data_augmentation',
)

# compute the mean and the variance for normalization
data_augmentation.layers[0].adapt(x_train)
```

#### 多层感知器


```python
def mlp(x, hidden_uints, dropout_rate):
    for uints in hidden_uints:
        x = layers.Dense(uints, activation = tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
```

#### 将patch创建实施为一层


```python
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes = [1, self.patch_size, self.patch_size,1],
            strides = [1,self.patch_size, self.patch_size, 1],
            rates = [1,1,1,1],
            padding='VALID',
        )
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
```

* 显示示例图像的patch


```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype('uint8'))
plt.axis('off')
plt.show()
```


![png](output_14_0.png)



```python
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size = (image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"image size: {image_size} x {image_size}")
print(f"Patch size: {patch_size} x {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
```

    image size: 72 x 72
    Patch size: 6 x 6
    Patches per image: 144
    Elements per patch: 108
    


```python
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype('uint8'))
    plt.axis('off')
```


![png](output_16_0.png)


#### 实现patch编码层

* PatchEncoder层通过将patch投影到大小为projection_dim的向量中来线性地对其进行变换。此外，它还向嵌入的向量中添加了可学习的位置。


```python
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units = projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )
        
    def call(self, patch):
        positions = tf.range(start=0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
```

#### 建立ViT模型

* ViT模型是由多个Transformer模块组成，这些模块使用layer.MultiHeadAttention层作为应用于补丁序列的自注意力机制。Transformer块产生一个[batch_size, num_patches, projection_dim]张量，该张量通过带有softmax的分类器头进行处理以产生最终的类概率输出。


```python
def create_vit_classifier():
    inputs = layers.Input(shape = input_shape)
    # augment data
    augmented = data_augmentation(inputs)
    # create patches
    patches = Patches(patch_size)(inputs)
    # encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # create multiple layers of the transformer block
    for _ in range(transformer_layers):
        # layer normalization 1
        x1 = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        # create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = projection_dim, dropout = 0.1
        )(x1, x1)
        # skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # mlp
        x3 = mlp(x3, hidden_uints=transformer_uints, dropout_rate=0.1)
        # skip connection 2
        encoded_patches = layers.Add()([x3, x2])
        
    # create a [batch_size, proprojection_dim] tensor
    representation = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # add mlp
    features = mlp(representationm, hidden_uints=mlp_head_uints,dropout_rate=0.5)
    # classify outputs
    logits = layers.Dense(num_classes)(features)
    # create the keras model
    model = keras.Model(inputs=inputs, outputs = logits)
    return model
```

#### 编译，训练和评估模型


```python
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=leanring_rate, weight_decay = weight_decay
    )
    
    model.compile(
        optimizer = optimizer,
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name = 'accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name = 'top-5-accuracy'),
        ],
    )
    
    checkpoint_filepath = '/ckeckpoint'
    checkpoint_callback = keras,callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor = 'val_accuracy',
        save_best_only = True,
        save_weights_only = True,
    )
    
    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = num_epochs,
        validation_split = 0.1,
        callbacks = [checkpoint_callback],
    )
    
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test,y_test)
    print('test accuracy: ', round(accuracy * 100, 2))
    
    return history
```


```python
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
```
