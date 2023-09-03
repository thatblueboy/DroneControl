import numpy as np
import pybullet as p
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ActionType, ObservationType
from gym import spaces
from typing import List, Union
from math import sin,asin,cos
import numpy as np

from .utils.PositionConstraint import PositionConstraint
import numpy as np
import pybullet as p
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ActionType, ObservationType
from gym import spaces
from typing import List, Union
from math import sin,asin
import numpy as np

from .utils.PositionConstraint import PositionConstraint


class ObstacleAviary(BaseSingleAgentAviary):

    CLOSE_TO_FINISH_REWARD = 5
    SUCCESS_REWARD = 1000
    COLLISION_PENALTY = -1000

    SUCCESS_EPSILON = 0.1

    MINOR_SAFETY_BOUND_RADIUS = 0.2
    MAJOR_SAFETY_BOUND_RADIUS = 0.1 
    COLLISION_BOUND_RADIUS = 0.07

    DISTANCE_PENALTY = 4
    MINOR_SAFETY_PENALTY = 1
    MAJOR_SAFETY_PENALTY = 5

    def __init__(self,
                 geoFence:PositionConstraint,
                 returnRawObservations:bool=False,
                 provideFixedObstacles:bool=False,
                 obstacles:Union[List[np.ndarray], None]=None,
                 minObstacles:int=2,
                 maxObstacles:int=7,
                 randomizeObstaclesEveryEpisode:bool=True,
                 fixedAltitude:bool=False,
                 episodeLength:int=2000,
                 showDebugLines:bool=False,
                 randomizeDronePosition:bool=False,
                 simFreq:int=240,
                 controlFreq:int=48,
                 gui:bool=False,
                 dynamicObstacles:bool=False,
                 movementType:int=1):


        assert minObstacles <= maxObstacles, "Cannot have fewer minObstacles than maxObstacles"

        self.provideFixedObstacles = provideFixedObstacles
        self.returnRawObservations = returnRawObservations

        self.fixedAltitude = fixedAltitude

        self.minObstacles = minObstacles
        self.maxObstacles = maxObstacles
        self.movementType = movementType
        self.episodeLength = episodeLength
        self.episodeStepCount = 0

        self.geoFence = geoFence
        self.dynamicObstacles = dynamicObstacles
        self.dynamicObstaclesList = []
        self.velocity = 0.005
        self.ObsInfo = {}
        self.VO_Reward = 0

        self.o1_obs = []
        self.o2_obs = []

        self.randomizeDronePosition = randomizeDronePosition
        self.randomizeObstaclesEveryEpisode = randomizeObstaclesEveryEpisode and not self.provideFixedObstacles

        self.targetPos = [self.geoFence.xmax - ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS, (self.geoFence.ymin + self.geoFence.ymax)/2, (self.geoFence.zmin + self.geoFence.zmax)/2]

        if not randomizeDronePosition:
            self.initPos = [self.geoFence.xmin + ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS, (self.geoFence.ymin + self.geoFence.ymax)/2, (self.geoFence.zmin + self.geoFence.zmax)/2]
        else:
            self._randomizeDroneSpawnLocation()
                            
        self.altitude = (self.geoFence.zmin + self.geoFence.zmax)/2

        self.showDebugLines = gui and showDebugLines

        self.trajectory = []
        self.noisyTrajectory = []

        self.obstacles = []
        self.totalTimesteps = 0

        self.simFreq = simFreq
        self.controlFreq = controlFreq
        self.aggregatePhysicsSteps = simFreq//controlFreq

        super().__init__(drone_model=DroneModel.CF2X,
                        initial_xyzs=np.array([self.initPos]),
                        initial_rpys=np.array([[0, 0, 0]]),
                        physics=Physics.PYB,
                        freq=self.simFreq,
                        aggregate_phy_steps=self.aggregatePhysicsSteps,
                        gui=gui,
                        record=False,
                        obs=ObservationType.KIN,
                        act=ActionType.VEL)

        if self.provideFixedObstacles:
            self.obstaclePositions = obstacles
        else:
            self._generateObstaclePositions()

        self.obstacleOffsetLine = None

    def _observationSpace(self):

        
        if self.returnRawObservations:
            
            if not self.fixedAltitude:
                obsLowerBound = np.array([self.geoFence.xmin, #x
                                          self.geoFence.ymin, #y
                                          self.geoFence.zmin, #z 
                                          self.geoFence.xmin, #xt
                                          self.geoFence.ymin, #yt
                                          self.geoFence.zmin, #zt
                                          self.geoFence.xmin, #xo
                                          self.geoFence.ymin, #yo
                                          self.geoFence.zmin, #zo
                                        ])

                obsUpperBound = np.array([self.geoFence.xmax, #x
                                          self.geoFence.ymax, #y
                                          self.geoFence.zmax, #z 
                                          self.geoFence.xmax, #xt
                                          self.geoFence.ymax, #yt
                                          self.geoFence.zmax, #zt
                                          self.geoFence.xmax, #xo
                                          self.geoFence.ymax, #yo
                                          self.geoFence.zmax, #zo
                                        ])
            else:
                obsLowerBound = np.array([self.geoFence.xmin, #x
                                          self.geoFence.ymin, #y
                                          self.geoFence.xmin, #xt
                                          self.geoFence.ymin, #yt
                                          self.geoFence.xmin, #xo
                                          self.geoFence.ymin, #yo
                                        ])

                obsUpperBound = np.array([self.geoFence.xmax, #x
                                          self.geoFence.ymax, #y
                                          self.geoFence.xmax, #xt
                                          self.geoFence.ymax, #yt
                                          self.geoFence.xmax, #xo
                                          self.geoFence.ymax, #yo
                                        ])
        else:
            if not self.fixedAltitude:
                obsUpperBound = np.array([self.geoFence.xmax - self.geoFence.xmin, #dxt
                                          self.geoFence.ymax - self.geoFence.ymin, #dyt
                                          self.geoFence.zmax - self.geoFence.zmin, #dzt
                                          self.geoFence.xmax - self.geoFence.xmin, #dxo
                                          self.geoFence.ymax - self.geoFence.ymin, #dyo
                                          self.geoFence.zmax - self.geoFence.zmin, #dzo
                                          0, #O1-x
                                          0, #01-y
                                          0, #02-x
                                          0, #02-y
                                          0, #01-vx
                                          0, #01-vy
                                          0, #02-vx
                                          0,#02-vy
                                        ])
                
            else:
                obsUpperBound = np.array([self.geoFence.xmax - self.geoFence.xmin, #dxt
                                          self.geoFence.ymax - self.geoFence.ymin, #dyt
                                          self.geoFence.xmax - self.geoFence.xmin, #dxo
                                          self.geoFence.ymax - self.geoFence.ymin, #dyo
                                          0, #O1-pos
                                          0, #01-vel
                                          0, #02-pos
                                          0, #02-vel
                                        ])
                obsLowerBound = -obsUpperBound

        return spaces.Box(low=obsLowerBound, high=obsUpperBound, dtype=np.float32)

    def _actionSpace(self):
        
        # [vx, vy, vz, v_mag] or [vx, vy, v_mag] for fixedAltitude
        actLowerBound = np.array([-1] * (3 if self.fixedAltitude else 4))
        actUpperBound = np.array([1] * (3 if self.fixedAltitude else 4))
        return spaces.Box(low=actLowerBound, high=actUpperBound, dtype=np.float32)

    def _computeObs(self):

        state = self._getDroneStateVector(0)
        pos = state[:3]

        offsetToTarget = self.targetPos - pos
        offsetToClosestObstacle = self._computeOffsetToClosestObstacle()

        if self.fixedAltitude:
            pos = pos[:2]
            offsetToTarget = offsetToTarget[:2]
            offsetToClosestObstacle = offsetToClosestObstacle[:2]

        if self.returnRawObservations:
            observation = np.concatenate([pos, pos + offsetToTarget, pos + offsetToClosestObstacle])
        else:
            observation = np.concatenate([offsetToTarget, offsetToClosestObstacle, self.o1_obs[0], self.o1_obs[1],self.o2_obs[0],self.o2_obs[1]])

        return observation

    def _computeProcessedObservation(self, rawObservation):

        if self.fixedAltitude:
            pos = rawObservation[0:2]
            targetPos = rawObservation[2:4]
            closestObstaclePos = rawObservation[4:6]
        else:
            pos = rawObservation[0:3]
            targetPos = rawObservation[3:6]
            closestObstaclePos = rawObservation[6:9]

        offsetToTarget = targetPos - pos
        offsetToClosestObstacle = closestObstaclePos - pos

        return np.concatenate([offsetToTarget, offsetToClosestObstacle])
    

    def reset(self):
        self.episodeStepCount = 0
        self.trajectory = []
        self.noisyTrajectory = []
        self.dynamicObstaclesList = []
        self.ObsInfo = {}
        self.obstacles = []
        self.offsetLine = None
        self.targetLine = None
        self.o1_obs = []
        self.o2_obs = []
        
        p.resetSimulation(physicsClientId=self.CLIENT)

        if self.randomizeDronePosition:
            self._randomizeDroneSpawnLocation()

        self._housekeeping()
        self._updateAndStoreKinematicInformation()

        p.addUserDebugPoints([self.targetPos], [np.array([0, 1, 0])], pointSize=10, physicsClientId=self.CLIENT)
        
        if self.randomizeObstaclesEveryEpisode:
            self._generateObstaclePositions()

        self._spawnObstacles()

        if self.showDebugLines:
            self._drawGeoFence()

        return self._computeObs()

    def checkVO(self,pos1,vel1,r1,pos2,vel2,r2,T=5):
        
        r = r2 + r1
        pos = pos2 - pos1
        vel = vel1 - vel2
        dist = pos - r
        if vel[0] == 0:
            m = 1000000
        else:
            m = vel[1]/vel[0]
        g,f = pos[0], pos[1]
        a = 1 + m**2
        b = -2*g -2*f*m
        c = g**2 + f**2 - r**2
        delta = b**2 - 4*a*c
        if np.linalg.norm(vel) > np.linalg.norm(dist)/T:
            if delta < 0:
                return False
            return True
        return False

    def moveObsSHM(self, Obstacle):
        obsPos, obsOrn = p.getBasePositionAndOrientation(Obstacle)
        y_initial = self.ObsInfo[Obstacle][0][1]
        x_initial = self.ObsInfo[Obstacle][0][0]
        amp = self.ObsInfo[Obstacle][2][0]
        orientation = self.ObsInfo[Obstacle][2][1]
        x,y,z = obsPos
        if orientation == 0:
            phi = asin(y_initial/0.5)
            y_new = 0.5*sin(0.015*self.totalTimesteps + phi)
            p.resetBasePositionAndOrientation(Obstacle,[x,y_new,z],obsOrn)
            velocity = 0.015*0.5*cos(0.015*self.totalTimesteps+phi)
            self.ObsInfo[Obstacle][1] = velocity
            self.ObsInfo[Obstacle][3] = [0,velocity]
            self.o1_obs = [[x,y_new], [0,velocity]]
        if orientation == 1:
            x_new = x_initial + 0.5*sin(0.015*self.totalTimesteps)
            p.resetBasePositionAndOrientation(Obstacle,[x_new,y,z],obsOrn)
            velocity = 0.015*0.5*cos(0.015*self.totalTimesteps)
            self.ObsInfo[Obstacle][1] = velocity
            self.ObsInfo[Obstacle][3] = [velocity,0]
            self.o1_obs = [[x_new,y], [velocity, 0]]
        
        
    def moveObsLinear(self, Obstacle):
        obsPos, obsOrn = p.getBasePositionAndOrientation(Obstacle)
        y_initial = self.ObsInfo[Obstacle][0][1]
        x_initial = self.ObsInfo[Obstacle][0][0]
        amp = self.ObsInfo[Obstacle][2][0]
        orientation = self.ObsInfo[Obstacle][2][1]
        x,y,z = obsPos

        if x > 0.1:
            x_new = x - self.velocity
            self.ObsInfo[Obstacle][3] = [-self.velocity,0]
        else:
            x_new = x
            self.ObsInfo[Obstacle][3] = [0,0]
        p.resetBasePositionAndOrientation(Obstacle,[x_new,y,z],obsOrn)
        self.o2_obs = [[x_new,y], self.ObsInfo[Obstacle][3]]


        """
        1) if steps = 0 set x coordinate as x=2
        2) when x = 0.1 stop
        DONT REVERSE VELOCITY
        MAKE SURE VELOCITY STARTS OUT NEGATIVE
        WHEN IT REACHES X = 0.1, VEL = 0
        """

        # if orientation == 0:
        #     velocity = self.ObsInfo[Obstacle][1]
        #     if (y>y_initial and abs(y-(y_initial+amp))<0.009) or (y<y_initial and abs(y-(y_initial-amp))<0.009):
        #         velocity = -1*velocity
        #         self.ObsInfo[Obstacle][1] = velocity
        #     y_new = y + velocity
        #     self.ObsInfo[Obstacle][3] = [0,velocity]
        #     p.resetBasePositionAndOrientation(Obstacle,[x,y_new,z],obsOrn)
        
        # if orientation == 1:
        #     velocity = self.ObsInfo[Obstacle][1]
        #     if (x>x_initial and abs(x-(x_initial+amp))<0.009) or (x<x_initial and abs(x-(x_initial-amp))<0.009):
        #         velocity = -1*velocity
        #         self.ObsInfo[Obstacle][1] = velocity
        #     x_new = x + velocity
        #     self.ObsInfo[Obstacle][3] = [velocity,0]
        #     p.resetBasePositionAndOrientation(Obstacle,[x_new,y,z],obsOrn)



    def step(self, action):
        self.VO_Reward = 0

        if len(self.dynamicObstaclesList) > 0:
            for Obstacle in self.dynamicObstaclesList:
                if self.ObsInfo[Obstacle][4] == 'shm':
                    self.moveObsSHM(Obstacle)
                if self.ObsInfo[Obstacle][4] == 'linear':
                    self.moveObsLinear(Obstacle)
                
        if self.fixedAltitude:
            action = np.insert(action, 2, 0)

        self.episodeStepCount += 1
        self.totalTimesteps += 1

        state = self._getDroneStateVector(0)
        pos = state[:3]
        self.trajectory.append(pos)

        if self.obstacleOffsetLine is not None:
            p.removeUserDebugItem(self.obstacleOffsetLine, physicsClientId=self.CLIENT)
        
        offsetToClosestObstacle = self._computeOffsetToClosestObstacle()

        if self.showDebugLines:
            if np.linalg.norm(offsetToClosestObstacle) < ObstacleAviary.MAJOR_SAFETY_BOUND_RADIUS:
                lineColor = np.array([1, 0, 0])
            elif np.linalg.norm(offsetToClosestObstacle) < ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS:
                lineColor = np.array([1, 1, 0])
            else:
                lineColor = np.array([0, 1, 0])

            self.obstacleOffsetLine = p.addUserDebugLine(pos, pos + offsetToClosestObstacle, lineColor)

            self._drawTrajectory()

        return super().step(action)

    def _computeReward(self):

        self.VO_Reward = 0
        state = self._getDroneStateVector(0)
        pos = state[:3]
        x,y = pos[0], pos[1]


        for obs in self.dynamicObstaclesList:
            velocity_vec_obs = self.ObsInfo[obs][3]
            obsPos, obsOrn = p.getBasePositionAndOrientation(obs)
            ox,oy,oz = obsPos
            drone_vel = state[9:12]
            vx,vy = drone_vel[0], drone_vel[1]

            if self.checkVO(np.array([x,y]),np.array([vx,vy]),0.1,np.array([ox,oy]),np.array(velocity_vec_obs),0.06, 3):
                self.VO_Reward += 0.1


        if np.linalg.norm(self.targetPos - pos) < ObstacleAviary.SUCCESS_EPSILON:
            return ObstacleAviary.SUCCESS_REWARD
        
        offsetToClosestObstacle = self._computeOffsetToClosestObstacle()
        
        distToClosestObstacle = np.linalg.norm(offsetToClosestObstacle)
        
        if distToClosestObstacle < ObstacleAviary.COLLISION_BOUND_RADIUS:
            return ObstacleAviary.COLLISION_PENALTY

        majorBoundBreach = distToClosestObstacle < ObstacleAviary.MAJOR_SAFETY_BOUND_RADIUS
        minorBoundBreach = distToClosestObstacle < ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS

        return  - ObstacleAviary.DISTANCE_PENALTY*np.linalg.norm(self.targetPos - pos) \
                - ObstacleAviary.MAJOR_SAFETY_PENALTY*majorBoundBreach \
                - ObstacleAviary.MINOR_SAFETY_PENALTY*minorBoundBreach \
                - self.VO_Reward

    def _computeOffsetToClosestObstacle(self):

        state = self._getDroneStateVector(0)
        pos = state[:3]
        x,y,z = pos

        # Check distance to all obstacles
        obstacleOffset = None
        
        for obstacle in self.obstacles:
            pointData = p.getClosestPoints(self.DRONE_IDS[0], obstacle, 100, -1, -1, physicsClientId=self.CLIENT)[0]
            offset = np.array(pointData[6]) - pos

            if obstacleOffset is None:
                obstacleOffset = offset
            else:
                obstacleOffset = min(obstacleOffset, offset, key=np.linalg.norm)

        # Check distance to boundaries
        # xBoundDist = x - self.geoFence.xmin
        xBoundDist = min(x - self.geoFence.xmin, self.geoFence.xmax - x)
        yBoundDist = min(y - self.geoFence.ymin, self.geoFence.ymax - y)
        zBoundDist = min(z - self.geoFence.zmin, self.geoFence.zmax - z) if not self.fixedAltitude else np.inf
        

        boundDists = [xBoundDist, yBoundDist, zBoundDist]

        # if xBoundDist == min(boundDists):
        #     fenceOffset = np.array([-(x - self.geoFence.xmin), 0, 0])

        if xBoundDist == min(boundDists):
            if x - self.geoFence.xmin < self.geoFence.xmax - x:
                fenceOffset = np.array([-(x - self.geoFence.xmin), 0, 0])
            else:
                fenceOffset = np.array([(self.geoFence.xmax - x), 0, 0])
        elif yBoundDist == min(boundDists):
            if y - self.geoFence.ymin < self.geoFence.ymax - y:
                fenceOffset = np.array([0, -(y - self.geoFence.ymin), 0])
            else:
                fenceOffset = np.array([0, (self.geoFence.ymax - y), 0])
        else:
            if z - self.geoFence.zmin < self.geoFence.zmax - z:
                fenceOffset = np.array([0, 0, -(z - self.geoFence.zmin)])
            else:
                fenceOffset = np.array([0, 0, (self.geoFence.zmax - z)])

        return fenceOffset if obstacleOffset is None else min(fenceOffset, obstacleOffset, key=np.linalg.norm) 

    def _computeDone(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]

        if self.episodeLength != -1 and self.episodeStepCount >= self.episodeLength:
            return True

        if np.linalg.norm(self.targetPos - pos) < 0.1:
            return True

        offsetToClosestObstacle = self._computeOffsetToClosestObstacle()

        if np.linalg.norm(offsetToClosestObstacle) <= ObstacleAviary.COLLISION_BOUND_RADIUS:
            return True

        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[:3]

        if self.episodeLength != -1 and self.episodeStepCount >= self.episodeLength:
            dist = np.linalg.norm(self.targetPos - pos)
            return {'success': False, 'reason': "outOfTime", 'dist': dist}

        if np.linalg.norm(self.targetPos - pos) < 0.1:
            return {'success': True}

        offsetToClosestObstacle = self._computeOffsetToClosestObstacle()

        if np.linalg.norm(offsetToClosestObstacle) <= ObstacleAviary.COLLISION_BOUND_RADIUS:
            return {'success': False, 'reason': "collision"}

        return {}

    
    # Utility Functions
    def _drawGeoFence(self):

        pc = self.geoFence
        p.addUserDebugLine([pc.xmin, pc.ymin, pc.zmin], [pc.xmax, pc.ymin, pc.zmin], lineWidth=3)
        p.addUserDebugLine([pc.xmin, pc.ymin, pc.zmin], [pc.xmin, pc.ymin, pc.zmax], lineWidth=3)
        p.addUserDebugLine([pc.xmin, pc.ymin, pc.zmin], [pc.xmin, pc.ymax, pc.zmin], lineWidth=3)

        p.addUserDebugLine([pc.xmax, pc.ymin, pc.zmax], [pc.xmin, pc.ymin, pc.zmax], lineWidth=3)
        p.addUserDebugLine([pc.xmax, pc.ymin, pc.zmax], [pc.xmax, pc.ymax, pc.zmax], lineWidth=3)
        p.addUserDebugLine([pc.xmax, pc.ymin, pc.zmax], [pc.xmax, pc.ymin, pc.zmin], lineWidth=3)

        p.addUserDebugLine([pc.xmin, pc.ymax, pc.zmax], [pc.xmin, pc.ymin, pc.zmax], lineWidth=3)
        p.addUserDebugLine([pc.xmin, pc.ymax, pc.zmax], [pc.xmax, pc.ymax, pc.zmax], lineWidth=3)
        p.addUserDebugLine([pc.xmin, pc.ymax, pc.zmax], [pc.xmin, pc.ymax, pc.zmin], lineWidth=3)

        p.addUserDebugLine([pc.xmax, pc.ymax, pc.zmin], [pc.xmax, pc.ymin, pc.zmin], lineWidth=3)
        p.addUserDebugLine([pc.xmax, pc.ymax, pc.zmin], [pc.xmin, pc.ymax, pc.zmin], lineWidth=3)
        p.addUserDebugLine([pc.xmax, pc.ymax, pc.zmin], [pc.xmax, pc.ymax, pc.zmax], lineWidth=3)

        for xlim in [pc.xmin, pc.xmax]:
            for ylim in [pc.ymin, pc.ymax]:
                for zlim in [pc.zmin, pc.zmax]:
                    p.addUserDebugText(f"({xlim}, {ylim}, {zlim})", np.array([xlim, ylim, zlim]), textSize=1)

    def _drawTrajectory(self):

        if len(self.trajectory) > 3:
            p.addUserDebugLine(self.trajectory[-2], self.trajectory[-1], lineColorRGB=[1, 0, 0], lineWidth=2)
        
        if len(self.noisyTrajectory):
            p.addUserDebugPoints([self.noisyTrajectory[-1]], [[0, 1, 0]], pointSize=2, physicsClientId=self.CLIENT)

    def _generateObstaclePositions(self):
        self.obstaclePositions = []
        
        if self.minObstacles >= 3:
            nObstacles = np.random.randint(2,6)
        else:
            nObstacles = np.random.randint(self.minObstacles, self.maxObstacles)
        for _ in range(nObstacles):
            # Position along all axes is uniform
            obstaclePos = self.geoFence.generateRandomPosition(padding=0.4)

            # Sample Y-axis position from normal distribution for more obstacles towards the middle of the path
            obstaclePos[1] = np.random.normal((self.geoFence.ymin + self.geoFence.ymax)/2, np.abs(self.geoFence.ymax)/3)
            obstaclePos[1] = np.clip(obstaclePos[1], self.geoFence.ymin, self.geoFence.ymax)
            
            # Sample Z-axis position from normal distribution for more obstacles towards the initial altitude of the drone
            obstaclePos[2] = np.random.random() * (self.geoFence.zmax - self.geoFence.zmin - 0.4) + (self.geoFence.zmin + 0.2)

            if self.fixedAltitude:
                obstaclePos[2] = self.altitude
            if _ == 1:
                obstaclePos[0] = 2
            
            self.obstaclePositions.append(obstaclePos)


    def _spawnObstacles(self):
        
        moving_count = 0
        for obstaclePos in self.obstaclePositions:
            currObstacle = p.loadURDF('sphere_small.urdf', obstaclePos, globalScaling=2)
            p.changeDynamics(currObstacle, -1, mass=0)
            self.obstacles.append(currObstacle)
            if self.dynamicObstacles and moving_count<2:
                self.dynamicObstaclesList.append(currObstacle)
                pos, orient = p.getBasePositionAndOrientation(currObstacle)
                self.ObsInfo[currObstacle] = [pos,self.velocity, [np.random.uniform(0.2,0.5),0 if currObstacle==2 else 1], None] #[pos, velocity_mag, [amp, movement_orientation], vel_vector]
                if self.movementType == 1:
                    if currObstacle == 2:
                        self.ObsInfo[currObstacle].append('shm')
                    if currObstacle == 3:
                        self.ObsInfo[currObstacle].append('linear')
                moving_count+=1

    def _randomizeDroneSpawnLocation(self):
        y_scale = self.geoFence.ymax - self.geoFence.ymin
        self.initPos = np.array([self.geoFence.xmin + ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS, 
                        (self.geoFence.ymin + self.geoFence.ymax) + np.random.uniform(-y_scale/2 + ObstacleAviary.COLLISION_BOUND_RADIUS*2, y_scale/2 - ObstacleAviary.COLLISION_BOUND_RADIUS*2),
                        (self.geoFence.zmin + self.geoFence.zmax)/2])

        self.INIT_XYZS = np.array([self.initPos])

