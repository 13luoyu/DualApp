import tensorflow as tf
from cnn_bounds_for_figure_3 import *
import os

def printlog(s):
    print(s, file=open("logs/figure_3_"+timestr+".txt", "a"))

if __name__ == '__main__':
    
    # epss = [0.5, 1, 1.5, 2, 2.5]

    for i in range(50000):
    
        net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-3,maxval=3),
            bias_initializer=tf.zeros_initializer()),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-3,maxval=3),
            bias_initializer=tf.zeros_initializer()),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-3,maxval=3),
            bias_initializer=tf.zeros_initializer())])

        X = tf.random.uniform((1,1,2), minval=-1, maxval=1).numpy()
        net(X)

        # 为权重赋值
        # optimizer2 = tf.keras.optimizers.SGD(lr=1.0)
        # weights = [[1,1],[1,-1]]
        # optimizer2.apply_gradients(zip([net.weights[0]-weights], net.layers[1].trainable_variables))
        # weights = [[1,-1],[3,2]]
        # optimizer2.apply_gradients(zip([net.weights[2]-weights], net.layers[3].trainable_variables))
        # weights = [[1,-2],[-1,1]]
        # optimizer2.apply_gradients(zip([net.weights[4]-weights], net.layers[5].trainable_variables))



        model_name = 'example'+str(i)+'_sigmoid'+'.h5'
        net.save(model_name)
        
        printlog("=========================================")
        printlog(model_name)
        printlog("Weights:")
        w0 = net.weights[0].numpy()
        printlog("{} {} {} {}".format(w0[0,0], w0[0,1], w0[1,0], w0[1,1]))
        # printlog("layer1.bias: {}".format(net.weights[1]))
        w1 = net.weights[2].numpy()
        printlog("{} {} {} {}".format(w1[0,0], w1[0,1], w1[1,0], w1[1,1]))
        # printlog("layer2.bias: {}".format(net.weights[3]))
        w2 = net.weights[4].numpy()
        printlog("{} {} {} {}".format(w2[0,0], w2[0,1], w2[1,0], w2[1,1]))

        X = np.array([[[0,0]]])

        # X = tf.random.uniform((1,1,2), minval=-1, maxval=1).numpy()
        LB_total, UB_total = run_certified_bounds(model_name, X, method='NeWise')
        printlog("NeWise:")
        printlog("x1:[{} {}]".format(LB_total[0][0][0][0],UB_total[0][0][0][0]))
        printlog("x2:[{} {}]".format(LB_total[0][0][0][1],UB_total[0][0][0][1]))
        printlog("x3:[{} {}]".format(LB_total[1][0][0][0],UB_total[1][0][0][0]))
        printlog("x4:[{} {}]".format(LB_total[1][0][0][1],UB_total[1][0][0][1]))
        printlog("x5:[{} {}]".format(LB_total[3][0][0][0],UB_total[3][0][0][0]))
        printlog("x6:[{} {}]".format(LB_total[3][0][0][1],UB_total[3][0][0][1]))
        printlog("y1:[{} {}]".format(LB_total[5][0][0][0],UB_total[5][0][0][0]))
        printlog("y2:[{} {}]".format(LB_total[5][0][0][1],UB_total[5][0][0][1]))
        printlog("\n")
        
        LB_total, UB_total = run_certified_bounds(model_name, X, method='DeepCert')
        printlog("DeepCert:")
        printlog("x1:[{} {}]".format(LB_total[0][0][0][0],UB_total[0][0][0][0]))
        printlog("x2:[{} {}]".format(LB_total[0][0][0][1],UB_total[0][0][0][1]))
        printlog("x3:[{} {}]".format(LB_total[1][0][0][0],UB_total[1][0][0][0]))
        printlog("x4:[{} {}]".format(LB_total[1][0][0][1],UB_total[1][0][0][1]))
        printlog("x5:[{} {}]".format(LB_total[3][0][0][0],UB_total[3][0][0][0]))
        printlog("x6:[{} {}]".format(LB_total[3][0][0][1],UB_total[3][0][0][1]))
        printlog("y1:[{} {}]".format(LB_total[5][0][0][0],UB_total[5][0][0][0]))
        printlog("y2:[{} {}]".format(LB_total[5][0][0][1],UB_total[5][0][0][1]))
        printlog("\n")
        
        LB_total, UB_total = run_certified_bounds(model_name, X, method='VeriNet')
        printlog("VeriNet:")
        printlog("x1:[{} {}]".format(LB_total[0][0][0][0],UB_total[0][0][0][0]))
        printlog("x2:[{} {}]".format(LB_total[0][0][0][1],UB_total[0][0][0][1]))
        printlog("x3:[{} {}]".format(LB_total[1][0][0][0],UB_total[1][0][0][0]))
        printlog("x4:[{} {}]".format(LB_total[1][0][0][1],UB_total[1][0][0][1]))
        printlog("x5:[{} {}]".format(LB_total[3][0][0][0],UB_total[3][0][0][0]))
        printlog("x6:[{} {}]".format(LB_total[3][0][0][1],UB_total[3][0][0][1]))
        printlog("y1:[{} {}]".format(LB_total[5][0][0][0],UB_total[5][0][0][0]))
        printlog("y2:[{} {}]".format(LB_total[5][0][0][1],UB_total[5][0][0][1]))
        printlog("\n")
        
        LB_total, UB_total = run_certified_bounds(model_name, X, method='RobustVerifier')
        printlog("RobustVerifier:")
        printlog("x1:[{} {}]".format(LB_total[0][0][0][0],UB_total[0][0][0][0]))
        printlog("x2:[{} {}]".format(LB_total[0][0][0][1],UB_total[0][0][0][1]))
        printlog("x3:[{} {}]".format(LB_total[1][0][0][0],UB_total[1][0][0][0]))
        printlog("x4:[{} {}]".format(LB_total[1][0][0][1],UB_total[1][0][0][1]))
        printlog("x5:[{} {}]".format(LB_total[3][0][0][0],UB_total[3][0][0][0]))
        printlog("x6:[{} {}]".format(LB_total[3][0][0][1],UB_total[3][0][0][1]))
        printlog("y1:[{} {}]".format(LB_total[5][0][0][0],UB_total[5][0][0][0]))
        printlog("y2:[{} {}]".format(LB_total[5][0][0][1],UB_total[5][0][0][1]))
        printlog("\n")
        printlog("=========================================")

        os.remove(model_name)
