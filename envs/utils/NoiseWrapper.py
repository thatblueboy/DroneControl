import numpy as np
import gym
from typing import Union, Tuple
from tabulate import tabulate
from gym import spaces

from .DenoiseEngines import LPFDenoiseEngine, KFDenoiseEngine
from ..ObstacleAviary import ObstacleAviary
#from ..MocapAviary import MocapAviary

class GaussianNoiseGenerator:

    def __init__(self, mu=0, sigma=1) -> None:
        self.mu = mu
        self.sigma = sigma

    def generateNoise(self, size=None) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, size=size)

    def __str__(self) -> str:
        return f"~N({self.mu}, {self.sigma})"

class NoiseWrapper(gym.Wrapper):

    def __init__(self, env:ObstacleAviary, mu:float, sigma:float, denoiseEngine:Union[None, LPFDenoiseEngine, KFDenoiseEngine]=None) -> None:

        super().__init__(env)
        self.denoiseEngine = denoiseEngine
        self.noiseGenerator = GaussianNoiseGenerator(mu, sigma)
        self.observation_space = self.newObservationSpace()

    def newObservationSpace(self):  

        obsUpperBound = np.array([self.geoFence.xmax - self.geoFence.xmin, #dxt
                                  self.geoFence.ymax - self.geoFence.ymin, #dyt
                                  self.geoFence.xmax - self.geoFence.xmin, #dxo
                                  self.geoFence.ymax - self.geoFence.ymin, #dyo
                                  0.5*self.omega2,
                                  self.vel1
                                ])
        obsLowerBound = -obsUpperBound

        return spaces.Box(low=obsLowerBound, high=obsUpperBound, dtype=np.float32)
    
    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        obs, reward, done, info = self.env.step(action)

        obs = self.corruptObservation(obs)

        self.noisyTrajectory.append(np.array([obs[0], obs[1], self.altitude]))
        if self.denoiseEngine is not None:
          self.denoiseEngine.reset(self.env.initPos)
          obs[:2] = self.denoiseEngine(obs[:2].copy(), np.zeros(2))
        obs = self.computeProcessedObservation(obs)

        # print("observation", obs)

        return obs, reward, done, info

    def computeVelocityFromAction(self, action):

        vel_dim = 2
        
        if np.linalg.norm(action[:vel_dim]) != 0:
            v_unit_vector = action[:vel_dim] / np.linalg.norm(action[:vel_dim])
        else:
            v_unit_vector = np.zeros(vel_dim)
        vel = self.env.SPEED_LIMIT * np.abs(action[-1]) * v_unit_vector

        return vel
    
    def computeProcessedObservation(self, rawObservation):
        pos = rawObservation[0:2]
        targetPos = rawObservation[2:4]
        closestObstaclePos = rawObservation[4:6]
        offsetToTarget = targetPos - pos
        offsetToClosestObstacle = closestObstaclePos - pos
        return np.concatenate([offsetToTarget, offsetToClosestObstacle, rawObservation[6:]])
    
    def corruptObservation(self, obs:np.ndarray) -> np.ndarray:
        noise = self.noiseGenerator.generateNoise(2)
        obs[:2] = obs[:2] + noise 
        return obs

    def reset(self) -> np.ndarray:
        obs = super().reset()
        obs = self.corruptObservation(obs)
        if self.denoiseEngine is not None:
            self.denoiseEngine.reset(self.env.initPos)
            obs[:2] = self.denoiseEngine(obs[:2].copy(), np.zeros(2))
        obs = self.computeProcessedObservation(obs)
        return obs
        
    def __str__(self) -> str:
        if self.env.randomizeObstaclesEveryEpisode:
            obstacleDetails = f"Random Obstacles per Episode ~ U({self.env.minObstacles}, {self.env.maxObstacles})"
        else:
            obstacleDetails = ', '.join([f"({x}, {y}, {z})" for x,y,z in self.env.obstacles])
        
        envDetails = {
            'Obstacles': obstacleDetails,
            'Fixed Altitude': self.env.fixedAltitude,
            'Noise': "None" if (self.noiseGenerator.mu, self.noiseGenerator.sigma) == (0, 0) else str(self.noiseGenerator), 
            'Denoiser': str(self.denoiseEngine),
        }

        return tabulate([(k,v) for (k,v) in envDetails.items()], tablefmt='pretty')
