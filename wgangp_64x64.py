# -*- coding: utf-8 -*-
"""
Training of the WGAN-GP model

Copyright (c) 2017 Ishaan Gulrajani
Copyright (c) 2018 Thomas Schlegl ... modified and extended

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import sys
import numpy as np
import re
sys.path.append(os.getcwd())

import tensorflow as tf
import time
import functools
from tqdm import tqdm # 进度条 tqdm 库

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.img_loader
import tflib.ops.layernorm
import tflib.plot


class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

timestamp = time.strftime("%Y-%m-%d-%H%M")# 定义生成文件夹wganTrain的时间戳
filename = os.path.basename(__file__).strip('.py')


MODE = 'wgan-gp'
RAND_SAMPLING = 'normal' # 'unif' 随机采样方式选择
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5
N_GPUS = 1
BATCH_SIZE = 64
LAMBDA = 10 # Gradient penalty hyperpar
OUTPUT_DIM = 64*64*1 # Number of pixels in each image
ZDIM = 128
TRAIN_EPOCHS = 7
ZSPACE_SMPL_NRIMG = 5
ZSPACE_SMPL_PTS = 13
checkpoint_iter = None

# 跑一次程序，生成一个文件夹
run_name = "%s_%s_crIt%d_%s" % (filename, RAND_SAMPLING, CRITIC_ITERS, timestamp)
checkpoint_dir = os.path.join("wganTrain", run_name, "checkpoints")
log_dir    = os.path.join("wganTrain", run_name, "logs")
samples_dir  = os.path.join("wganTrain", run_name, "samples")
z_interp_dir = os.path.join("wganTrain", run_name, "z_interp")

print ("执行到这里～～～～")

print bcolors.GREEN + "\n=== WGAN-GP TRAINING PARAMETERS ===" + bcolors.ENDC
lib.print_model_settings(locals().copy())

# DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

DEVICES = ['/gpu:0', '/gpu:1']
# DEVICES = ['/cpu:0']
print ("len(DEVICES):", len(DEVICES))
print (DEVICES)

# 归一化
def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        # 层归一化
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        # 批量归一化
        return lib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)

def my_Normalize(name, inputs, is_training):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return tf.layers.batch_normalization(inputs, axis=1, training=is_training, name=name)

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], 1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, is_training=True, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if is_training is not None:
        output = my_Normalize(name+'.BN1', output, is_training)
    else:
        output = Normalize(name+'.BN1', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if is_training is not None:
        output = my_Normalize(name+'.BN2', output, is_training)
    else:
        output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def GoodGenerator(n_samples, noise=None, rand_sampling=RAND_SAMPLING, dim=DIM, nonlinearity=tf.nn.relu, z_out=False, is_training=True, reuse=None):
    # 变量对于使用 TensorFlow 进行深度学习是至关重要的，因为模型的参数就是变量。
    # 在训练期间，你希望通过梯度下降在每个步骤更新参数；
    # 但在评估时，你希望保持参数不变，并将大量不同的测试集输入模型。
    # 通常，模型所有可训练参数都是变量。

    with tf.variable_scope('Generator', reuse=reuse):
        if noise is None:
            if rand_sampling == 'unif': # 均匀
                noise = tf.random_uniform([n_samples, ZDIM], minval=-1., maxval=1.)
            elif rand_sampling == 'normal': # 正态
                noise = tf.random_normal([n_samples, ZDIM])

        output = lib.ops.linear.Linear('Generator.Input', ZDIM, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])

        output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, is_training=is_training, resample='up')

        if is_training is not None:
            output = my_Normalize('Generator.OutputN', output, is_training)
        else:
            output = Normalize('Generator.OutputN', [0,2,3], output)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 1, 3, output)
        output = tf.tanh(output)

    if z_out:
        return tf.reshape(output, [-1, OUTPUT_DIM]), noise
    else:
        return tf.reshape(output, [-1, OUTPUT_DIM])


def GoodDiscriminator(inputs, dim=DIM, is_training=False, reuse=None, out_feats=True):
    with tf.variable_scope('Discriminator', reuse=reuse):
        output = tf.reshape(inputs, [-1, 1, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 1, dim, 3, output, he_init=False)

        output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, is_training=is_training, resample='down')

        output = tf.reshape(output, [-1, 4*4*8*dim])
        out_features = output
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    if out_feats:
        return tf.reshape(output, [-1]), out_features
    else:
        return tf.reshape(output, [-1])

# 深度学习中，模型训练一般都需要很长的时间，由于很多原因，导致模型中断训练，下面是继续断点训练的方法：save（）和load（）。
# 保存变量
def save(session, saver, checkpoint_dir, step):
    print(" [*] Saving checkpoint (step %d) ..." % step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(session, os.path.join(checkpoint_dir, "%s.model" % MODE), global_step=step)# 这里的step是循环训练的次数，也就是第几次迭代


def load(session, saver, checkpoint_dir, checkpoint_iter=None):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)# 注意此处是checkpoint存在的目录
    if ckpt and ckpt.model_checkpoint_path:
        if checkpoint_iter is not None:
            last_ckpt_epoch = re.match(r'.*.model-(\d+)', ckpt.model_checkpoint_path).group(1)
            target_ckpt_path = re.sub(last_ckpt_epoch, str(checkpoint_iter), ckpt.model_checkpoint_path)
            saver.restore(session, target_ckpt_path)
            idxx = target_ckpt_path.rfind('/')
            ckpt_name = target_ckpt_path[idxx+1:]
        else:
            saver.restore(session, ckpt.model_checkpoint_path)# 自动恢复model_checkpoint_path保存模型一般是最新
            idxx = ckpt.model_checkpoint_path.rfind('/')
            ckpt_name = ckpt.model_checkpoint_path[idxx+1:]
        return True, ckpt_name
    else:
        return False, ''

# 训练函数
def train():
    print ("执行到这里～～～～")

    Generator, Discriminator = GoodGenerator, GoodDiscriminator

    for dir_path in [checkpoint_dir, log_dir, samples_dir, z_interp_dir]:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    # 自动选择运行设备 ： tf.ConfigProto(allow_soft_placement=True)
    # 在tf中，通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。
    # 如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，
    # 为了防止这种情况，可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        print ("执行到这里～～～～")

        # placeholder：占位符，用来接收外部输入的节点
        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1, 64, 64])

        # Python startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。
        # 如果参数 beg 和 end 指定值，则在指定范围内检查。
        if tf.__version__.startswith('1.'): # tf的版本号是1.x开头的，我这里用的是'1.10.0'
            # 把一个张量划分成几个子张量
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))# len(DEVICES) 准备切成几份
            print ("\n\nDEVICES: %s\n\n"%DEVICES)
        else:
            split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)

        gen_costs, disc_costs = [], []

        # zip 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            # with tf.device('/cpu:0'): 指定使用GPU或CPU
            with tf.device(device):
                # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
                # tf.reshape()为了功能改变张量（tensor）的形状。shape传入一个向量，代表新tensor的维度数和每个维度的长度。
                real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
                fake_data = Generator(BATCH_SIZE/len(DEVICES), rand_sampling=RAND_SAMPLING, is_training=True)

                disc_real = Discriminator(real_data, out_feats=False)
                disc_fake = Discriminator(fake_data, reuse=True, out_feats=False)

                if MODE == 'wgan-gp':
                    # tf.reduce_mean 计算tensor（图像）的平均值。
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE/len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data

                    interpolates = real_data + (alpha*differences)

                    gradients = tf.gradients(Discriminator(interpolates, reuse=True, out_feats=False), [interpolates])[0]

                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

                    gradient_penalty = tf.reduce_mean((slopes-1.)**2)

                    disc_cost += LAMBDA * gradient_penalty

                # Python列表append()方法用于将传入的对象附加(添加)到现有列表中。
                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if MODE == 'wgan-gp':
            t_vars = tf.trainable_variables()
            gen_vars = [var for var in t_vars if 'Generator' in var.name]
            dis_vars = [var for var in t_vars if 'Discriminator' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_gen = [var for var in update_ops if 'Generator' in var.name] 
            update_ops_dis = [var for var in update_ops if 'Discriminator' in var.name] 

            with tf.control_dependencies(update_ops_gen):
                gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                              var_list=gen_vars, colocate_gradients_with_ops=True)
            with tf.control_dependencies(update_ops_dis):
                disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                               var_list=dis_vars, colocate_gradients_with_ops=True)

        # For generating samples
        if RAND_SAMPLING == 'unif':
            fixed_noise = tf.constant(np.random.uniform(-1, 1, size=(BATCH_SIZE, ZDIM)).astype('float32'))
        elif RAND_SAMPLING == 'normal':
            fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, ZDIM)).astype('float32'))

        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE / len(DEVICES)
            all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples], rand_sampling=RAND_SAMPLING, is_training=False, reuse=True ) )
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, 0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

        def generate_image(epoch, iteration):
            samples = session.run(all_fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            lib.save_images.save_images(samples.reshape((BATCH_SIZE, 1, 64, 64)), '{}/samples_epoch{}-{}.png'.format(samples_dir, epoch, iteration))

        # Dataset iterator
        train_gen, _ = lib.img_loader.load(BATCH_SIZE, 'wgan_train')

        nr_training_samples = lib.img_loader.get_nr_training_samples(BATCH_SIZE)
        nr_iters_per_epoch = nr_training_samples//BATCH_SIZE

        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images

        # Save a batch of ground-truth samples
        _x = inf_train_gen().next()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE/N_GPUS]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((BATCH_SIZE/N_GPUS, 1, 64, 64)), '{}/samples_groundtruth.png'.format(samples_dir))

        # EVALUATION: z-interpolation ******
        eval_query_noise = tf.placeholder(tf.float32, shape=[ZSPACE_SMPL_PTS, ZDIM])
        zeval_gen_imgs = Generator(ZSPACE_SMPL_PTS, noise=eval_query_noise, rand_sampling=RAND_SAMPLING, is_training=False, reuse=True )

        def get_z_interpolations(smpl_pts, z_dim=ZDIM, v_len_lim=0.5):
            z_samples = np.zeros((smpl_pts, z_dim), dtype=np.float32)

            v_max = np.ones((1, 100))*2  # *2 ... => 2 = [-1..1]
            v_len_max = np.sqrt((v_max**2).sum()) # vector_length of v_max
            v_len_limes = v_len_max * v_len_lim
            v_len = 0

            while v_len < v_len_limes:
                if RAND_SAMPLING == 'unif':
                    z_p1 = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
                    z_p2 = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
                elif RAND_SAMPLING == 'normal':
                    z_p1 = np.random.normal(size=(1, ZDIM)).astype('float32')
                    z_p2 = np.random.normal(size=(1, ZDIM)).astype('float32')

                v = z_p2 - z_p1
                v_len = np.sqrt((v**2).sum()) # vector_length of v

            steps = np.linspace(0., 1., smpl_pts)
            for i, s in enumerate(steps):
                z_samples[i, :] = z_p1 + s*v

            z_imgs = session.run(zeval_gen_imgs, feed_dict={eval_query_noise: z_samples})
            return ((z_imgs+1.)*(255.99/2)).astype('int32')

        saver = tf.train.Saver(max_to_keep=10)
        session.run(tf.global_variables_initializer())
        isLoaded, ckpt = load(session, saver, checkpoint_dir, checkpoint_iter)
        start_iter = 0

        # Train loop
        iteration = 0
        for epoch in tqdm(xrange(TRAIN_EPOCHS)):
            gen = inf_train_gen()
            while iteration < ((epoch+1)*(nr_iters_per_epoch-CRITIC_ITERS)):
                start_time = time.time()

                # -- TRAIN generator --
                if (iteration+1) > 1:
                    _gen_cost, _ = session.run([gen_cost, gen_train_op])

                # -- TRAIN critic --
                disc_iters = CRITIC_ITERS
                for i in xrange(disc_iters):
                    _data = gen.next()
                    iteration += 1
                    _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

                #  -- LOGGING **
                lib.plot.tickit(iteration)

                if (iteration == (3*disc_iters)) or (iteration % (100*disc_iters) == 0):
                    lib.plot.plot('train disc cost', _disc_cost)
                    lib.plot.plot('train gen cost', _gen_cost)
                    lib.plot.plot('time', time.time() - start_time)

                if (iteration == (10*disc_iters)) or (iteration == (100*disc_iters)) or (iteration % (1000*disc_iters) == 0):
                    generate_image(epoch+1, iteration)

                if (iteration < 10) or (iteration % (100*disc_iters) == 0):
                    lib.plot.flush(log_dir)

                    total_samples_seen = iteration * BATCH_SIZE
                    if (epoch+1)==1:
                        nr_samples_within_epoch = total_samples_seen
                    else:
                        nr_samples_within_epoch = np.mod(total_samples_seen, epoch*(nr_iters_per_epoch-CRITIC_ITERS))
                    
                    print bcolors.GREEN + "\tSaw real samples of %d full epochs and %10d additinal samples .. " \
                          %(epoch, nr_samples_within_epoch) + \
                                "(%.3f%% of epoch done!)" \
                                    %(float(nr_samples_within_epoch)/nr_training_samples) + bcolors.ENDC

                if (iteration == (100*disc_iters)) or (iteration == (start_iter+(1000*disc_iters))) or (iteration % (1000*disc_iters) == 0):
                    z_imgs_out = np.zeros((ZSPACE_SMPL_PTS, 64*ZSPACE_SMPL_NRIMG,64), dtype=np.int32)
                    for _zi in range(ZSPACE_SMPL_NRIMG):
                        z_space_smpls = get_z_interpolations(ZSPACE_SMPL_PTS)
                        z_imgs_out[:, _zi*64:(_zi+1)*64, :] = z_space_smpls.reshape(ZSPACE_SMPL_PTS, 64, 64)
                    lib.save_images.save_images_as_row(z_imgs_out, os.path.join(z_interp_dir, 'z_smpls-epoch%d-%05d.png' % (epoch+1, iteration)) )

            print bcolors.BLUE + "\nEND OF EPOCH - SAVING CHECKPOINT\n" + bcolors.ENDC
            save(session, saver, checkpoint_dir, epoch+1)
            generate_image(epoch+1, iteration)

        # SAVE FINAL MODEL
        save(session, saver, checkpoint_dir, epoch+1)
        print bcolors.BLUE + "\nSAVING CHECKPOINT" + bcolors.ENDC
        print bcolors.BLUE + "Training done!\n" + bcolors.ENDC

# if __name__ == '__main__简单的理解就是，如果我们执行本模块，那么代码块被执行运行。
# 如果被导入到其他模块，则处于name中的代码块不执行，这是一种编码习惯....
if __name__ == '__main__':
    train()

