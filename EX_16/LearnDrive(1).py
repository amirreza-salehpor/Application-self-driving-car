import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import gym
from gym_donkeycar.envs import donkey_env
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
import pyasyncore

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_data(N, Nt):
    images = []
    dflabels = pd.read_json('Record/record_0.json', lines=True)
    for i in range(0,N):
        image_data=cv2.imread('Record/'+str(i)+'_cam-image_array_.jpg')/255
        image_data = rgb2gray(image_data)
        image_data = cv2.resize(image_data, (80,80))
        images.append(image_data)
        if i>0:
            dflabels = pd.concat([dflabels, pd.read_json('Record/record_'+str(i)+'.json', lines=True)], ignore_index=True)
    ind = np.arange(0,N)
    sample_test = np.random.choice(ind, Nt, replace=False)
    sample_train = np.setdiff1d(ind, sample_test)
    labels_test = []
    labels_train = []
    for i in sample_test:
        labels_test.append(dflabels['user/angle'][i])
    for i in sample_train:
        labels_train.append(dflabels['user/angle'][i])
    images_test = []
    images_train = []
    for i in sample_test:
        images_test.append(images[i])
    for i in sample_train:
        images_train.append(images[i])

    images_test = np.reshape(images_test, [Nt,80,80,1])
    images_train = np.reshape(images_train, [N-Nt,80,80,1])
    labels_test = np.array(labels_test)
    labels_test = np.reshape(labels_test, [Nt,1])
    labels_train = np.array(labels_train)
    labels_train = np.reshape(labels_train, [N-Nt,1])
    return images_train, labels_train, images_test, labels_test

#%%
def create_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(80,80,1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='linear'),
            tf.keras.layers.Dense(1, activation='linear'), 
            ])

    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    return model

def train_model(model, images_train, labels_train, images_test, labels_test):
    model.fit(images_train, labels_train, validation_data=(images_test,labels_test), epochs=10, batch_size=20) #, callbacks=callbacks
    model.save_weights('LearnDrive.h5')
    return model

def drive(state, model):
    statemod = np.reshape(state, [1,80,80,1])
    return model.predict(statemod)[0]

if __name__ == "__main__":
    N=673
    Nt=50
    TRAIN = True
    DRIVE = False
    EPISODES = 1
    MAX_SPEED = 6
    model=create_model()
    if TRAIN:
        images_train, labels_train, images_test, labels_test = load_data(N,Nt)
        model = train_model(model, images_train, labels_train, images_test, labels_test)
    else:
        model.load_weights('LearnDrive.h5')
    if DRIVE:
        env = gym.make("donkey-generated-roads-v0")
        for e in range(EPISODES):
            state = env.reset()/255
            state = env.reset()/255
            state = rgb2gray(state)
            state = cv2.resize(state, (80,80))
            state = np.reshape(state, (80,80,1))
            score = 0
            thr = 1
            for t in range(0,1000000):
                action = model.predict(state.reshape(1,80,80,1))[0][0]
                print(action)
                next_state, reward, done, info = env.step([action, thr])
                thr = np.clip(MAX_SPEED - info['speed'], 0, 1)
                next_state = next_state/255
                next_state = rgb2gray(next_state)
                next_state = cv2.resize(next_state, (80,80))
                next_state = np.reshape(next_state, (80,80,1))
                score += reward
                state = next_state
                if (done or t==999999):
                    print(e, score, t)                
                    break
        env.close()
