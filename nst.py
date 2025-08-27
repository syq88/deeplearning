
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# 关键：TF2 下切到 TF1 风格
tf.compat.v1.disable_eager_execution()

import nst_utils as nst  
%load_ext autoreload
%autoreload 2

# 可调超参
CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2),
]
ALPHA = 1.0         # 内容损失权重
BETA = 1000.0       # 风格损失权重
LR = 2.0            # Adam 学习率
NUM_ITERS = 180     # 迭代步数
PRINT_EVERY = 50    # 打印/保存频率

# ---------- 基础工具 ----------
def load_and_resize(path, width, height):
    """读图并缩放到 (width, height)，返回 float32 [H,W,C]"""
    img = Image.open(path).convert('RGB').resize((width, height), Image.LANCZOS)
    return np.array(img).astype(np.float32)

def gram_matrix(A):
    """
    A: [N, M] where N = channels, M = H*W
    return: [N, N]
    """
    return tf.matmul(A, A, transpose_b=True)

def compute_content_cost(a_C, a_G):
    """
    a_C, a_G: tensors of shape [1, n_H, n_W, n_C]
    """
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    # 1/(4*H*W*C) * ||a_C - a_G||^2
    return tf.reduce_sum(tf.square(a_C - a_G)) / (4.0 * n_H * n_W * n_C)

def compute_layer_style_cost(a_S, a_G):
    """
    a_S, a_G: [1, H, W, C]
    J_style_layer = 1/(4*C^2*(H*W)^2) * ||G_S - G_G||^2
    """
    _, H, W, C = a_G.get_shape().as_list()

    a_S_unrolled = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), [C, -1])
    a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), [C, -1])

    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    norm = 4.0 * (C ** 2) * ((H * W) ** 2)
    return tf.reduce_sum(tf.square(GS - GG)) / norm

def compute_style_cost(model, style_image, style_layers):
    """
    汇总多层风格损失（加权和）
    """
    J_style = 0
    with tf.compat.v1.Session() as s_tmp:
        s_tmp.run(tf.compat.v1.global_variables_initializer())
        s_tmp.run(model['input'].assign(style_image))
        # 先把每一层的风格特征取出来做成 tf.constant，冻结为目标
        style_targets = {}
        for layer_name, coeff in style_layers:
            a_S = s_tmp.run(model[layer_name])
            style_targets[layer_name] = tf.constant(a_S)
    # 用当前图里的对应层输出来对比
    for layer_name, coeff in style_layers:
        a_S_const = style_targets[layer_name]
        a_G = model[layer_name]
        J_style += coeff * compute_layer_style_cost(a_S_const, a_G)
    return J_style

# ---------- 主流程 ----------
def neural_style_transfer(
    content_path=nst.CONFIG.CONTENT_IMAGE,
    style_path=nst.CONFIG.STYLE_IMAGE,
    output_dir=nst.CONFIG.OUTPUT_DIR,
    num_iters=NUM_ITERS,
    alpha=ALPHA,
    beta=BETA,
    lr=LR,
    print_every=PRINT_EVERY,
):
    os.makedirs(output_dir, exist_ok=True)

    # 读入并缩放图像到与 VGG 输入一致的大小
    H, W = nst.CONFIG.IMAGE_HEIGHT, nst.CONFIG.IMAGE_WIDTH
    content_raw = load_and_resize(content_path, W, H)
    style_raw   = load_and_resize(style_path,   W, H)

    # 预处理（减均值 + 扩一维 batch）
    content_image = nst.reshape_and_normalize_image(content_raw)
    style_image   = nst.reshape_and_normalize_image(style_raw)

    # 构建 VGG 图（权重常量化，输入是唯一可训练变量）
    model = nst.load_vgg_model(nst.CONFIG.VGG_MODEL)

    with tf.compat.v1.Session() as s_tmp:
        s_tmp.run(tf.compat.v1.global_variables_initializer())
        s_tmp.run(model['input'].assign(content_image))
        a_C = s_tmp.run(model[CONTENT_LAYER])
    a_C_const = tf.constant(a_C)

    J_style = compute_style_cost(model, style_image, STYLE_LAYERS)

    a_G_content = model[CONTENT_LAYER]
    J_content = compute_content_cost(a_C_const, a_G_content)


    J_total = alpha * J_content + beta * J_style

    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(J_total, var_list=[model['input']])


    generated_image = nst.generate_noise_image(content_image)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(model['input'].assign(generated_image))

        for i in range(1, num_iters + 1):
            sess.run(train_step)

            if i % print_every == 0 or i == 1 or i == num_iters:
                Jt, Jc, Js = sess.run([J_total, J_content, J_style])
                print(f"Iter {i:04d} | total: {Jt:.2f} | content: {Jc:.2f} | style: {Js:.2f}")

                out = sess.run(model['input'])

        final = sess.run(model['input'])
        out_path = os.path.join(output_dir, "final2.png")
        nst.save_image(out_path, final)
        print(f"完成！结果已保存：{out_path}")

if __name__ == "__main__":
    neural_style_transfer( content_path="images/.jpg",style_path="images/.jpg",output_dir="output")#输入你的照片
