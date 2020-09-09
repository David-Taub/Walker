import numpy as np


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process
    """

    def __init__(self, output_size, mean=0, std_deviation=0.3, theta=1, dt=0.01):
        self.theta = theta
        self.mean = mean * np.ones(output_size)
        self.previous_x = mean
        self.std_dev = std_deviation
        self.dt = dt

    def __call__(self):

        # x = (self.previous_x + self.theta * (self.mean - self.previous_x) * self.dt + (
        #     self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)))
        x = self.previous_x + self.dt * (
            self.theta * (self.mean - self.previous_x) + self.std_dev * np.random.normal(size=self.mean.shape))
        self.previous_x = x
        return x


class MarkovSaltPepperNoise:
    def __init__(self, output_size, salt_to_pepper=0.02, pepper_to_salt=0.1):
        self.salt_to_pepper = salt_to_pepper
        self.pepper_to_salt = pepper_to_salt
        self.noise = np.ones(output_size)

    # def _reward_to_probabilty(self, reward):
    #     MAX_PROB = 0.05
    #     MIN_PROB = 0.001
    #     MIN_REWARD = 1
    #     MAX_REWARD = 7
    #     reward = min(MAX_REWARD, max(reward, MIN_REWARD))
    #     reward_streched = (reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
    #     return MIN_PROB + (1 - reward_streched) * (MAX_PROB - MIN_PROB)

    def __call__(self):
        # self.salt_to_pepper = self._reward_to_probabilty(reward)
        switch_salt_to_pepper = np.random.binomial(1, self.salt_to_pepper, self.noise.shape)
        switch_pepper_to_salt = np.random.binomial(1, self.pepper_to_salt, self.noise.shape)
        self.noise[(self.noise == 1) & (switch_salt_to_pepper == 1)] = np.random.uniform(low=-1.5, high=-0.5)
        self.noise[(self.noise != 1) & (switch_pepper_to_salt == 1)] = 1
        return self.noise
