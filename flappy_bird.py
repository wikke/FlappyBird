import random
from collections import deque
import numpy as np

import cv2
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Model, Input

import sys
sys.path.append("game/")
import wrapped_flappy_bird as flappy_bird

EPSILON = 1.0
# EPSILON_DECAY = 0.95 # 总结尝试得到的经验值
EPSILON_DECAY = 0.9 # 总结尝试得到的经验值
EPSILON_MIN = 0.2 # 我的经验值

GAMMA = 0.9

BATCH_SIZE = 32

ACTIONS = 2
MAX_MEMORY = 200
OBSERVATION_NUM = 200

HEIGHT = 80
WIDTH = 40
CHANNEL = 1

class FlappyBird():
    def __init__(self):
        self.step = 0
        self.last_image_data = None
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = self._build_model()
        self.model.summary()
        self.game = flappy_bird.GameState()

    def _build_model(self):
        ipt = Input(shape=(HEIGHT, WIDTH, CHANNEL))
        x = ipt
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = BatchNormalization()(x)
        # GlobalMaxPooling2D
        x = Flatten()(x)

        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(ACTIONS, activation='softmax')(x)
        model = Model(inputs=[ipt], outputs=[x])
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return model

    def _get_action(self, image_data, exploitation=False):
        global EPSILON

        exploration = not exploitation and (self.step <= OBSERVATION_NUM or np.random.uniform() < EPSILON)

        action = np.zeros(ACTIONS)
        if exploration:
            action[random.randint(0, ACTIONS-1)] = 1
        else:
            pred_action = self.model.predict(np.expand_dims(image_data, axis=0))[0]
            action[np.argmax(pred_action)] = 1

        if self.step > OBSERVATION_NUM and EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        return action

    def episode(self):
        print('Start a episode')

        episode_steps = 0
        while True:
            action = self._get_action(self.last_image_data)

            image_data, reward, terminal = self.game.frame_step(action)
            image_data = cv2.cvtColor(cv2.resize(image_data, (WIDTH, HEIGHT)), cv2.COLOR_BGR2GRAY)
            _, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
            image_data = np.expand_dims(image_data, axis=-1)
            self.memory.append((self.last_image_data, action, reward, image_data, terminal))

            self.step += 1
            episode_steps += 1
            self.last_image_data = image_data

            if terminal:
                break

        print('Episode tries {} steps'.format(episode_steps))

    def replay(self):
        if self.step <= OBSERVATION_NUM:
            return

        print('Replaying')

        batches = random.sample(self.memory, BATCH_SIZE)
        X = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))
        y = np.zeros((BATCH_SIZE, ACTIONS))

        for i in range(BATCH_SIZE):
            image_data, action, reward, next_image_data, terminal = batches[i]

            if not terminal:
                next_action_pred = self.model.predict(np.expand_dims(next_image_data, axis=0))[0]
                reward += GAMMA * np.max(next_action_pred)

            pred_action = self.model.predict(np.expand_dims(image_data, axis=0))[0]
            pred_action[np.argmax(action)] = reward

            X[i] = image_data
            y[i] = pred_action

        self.model.train_on_batch(X, y)

    def demo(self):
        if self.step <= OBSERVATION_NUM:
            return

        print('Start a Demo')

        demo_steps = 0
        last_image_data = None
        action = np.zeros(ACTIONS)
        action[random.randint(0, ACTIONS-1)] = 1

        while True:
            image_data, reward, terminal = self.game.frame_step(action)
            demo_steps += 1

            if terminal:
                break

            image_data = cv2.cvtColor(cv2.resize(image_data, (WIDTH, HEIGHT)), cv2.COLOR_BGR2GRAY)
            _, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
            image_data = np.expand_dims(image_data, axis=-1)

            action = self._get_action(image_data, exploitation=True)

        print('Demo tries {} steps'.format(demo_steps))

def main():
    bird = FlappyBird()

    for i in range(100000):
        bird.episode()
        bird.replay()

        if (i+1) % 1 == 0:
            bird.demo()

if __name__ == '__main__':
    main()
