import numpy as np
import pybullet as p
import pybullet_data
import os

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

from PIL import Image


from .utils.PositionConstraint import PositionConstraint


class ObstacleAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly a drone in a constrained space with obstacles.
    0-2 dynamic obstacles, one with linear motion,
    other with SHM motion. Default 2"""

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
                 provideFixedObstacles:bool=False,
                 obstacles:Union[List[np.ndarray], None]=None,
                 minObstacles:int=2, #unused
                 maxObstacles:int=7, #unused
                 randomizeObstaclesEveryEpisode:bool=True,
                 fixedAltitude:bool=True,
                 episodeLength:int=1000,
                 showDebugLines:bool=False,
                 randomizeDronePosition:bool=False,
                 simFreq:int=240,
                 controlFreq:int=48,
                 gui:bool=False,
                 dynamicObstacles:bool=True,
                 movementType:int=1):

        # assert minObstacles <= maxObstacles, "Cannot have fewer minObstacles than maxObstacles"
        #instead, assert if number of positions in obstacles is not 2
        if provideFixedObstacles:
            assert len(obstacles) == 2, "Must provide 2 fixed obstacles"
        self.provideFixedObstacles = provideFixedObstacles

        self.fixedAltitude = fixedAltitude

        self.minObstacles = minObstacles #unused
        self.maxObstacles = maxObstacles #unused
        self.movementType = movementType #currently only 1 type
        self.episodeLength = episodeLength
        self.episodeStepCount = 0

        self.geoFence = geoFence
        

        ############### MOVING OBSTACLES ###################
        self.dynamicObstacles = dynamicObstacles
        self.dynamicObstaclesList = []
        self.velocity = 0.005
        self.obsInfo = {} #dictionary, obstacle: [pos, vel, initial position, movementType]
        self.VO_Reward = 0

        # self.o1_obs = []
        # self.o2_obs = []
        #MOVEMENT TYPE 1
        self.vel1 = 0.05; #linear velocity of obstacle 1
        self.omega2 = 0.05 #angular velocity of obstacle 2
        ####################################################

        self.simulationTime = 0

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
                        physics=Physics.PYB, #OTHER MODES NOT ACCOUNTED FOR
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

        obsLowerBound = np.array([self.geoFence.xmin, #x
                                    self.geoFence.ymin, #y
                                    self.geoFence.xmin, #xt
                                    self.geoFence.ymin, #yt
                                    self.geoFence.xmin, #xo
                                    self.geoFence.ymin, #yo
                                    0.5*self.omega2,
                                    self.vel1
                                       ])

        obsUpperBound = np.array([self.geoFence.xmax, #x
                                  self.geoFence.ymax, #y
                                  self.geoFence.xmax, #xt
                                  self.geoFence.ymax, #yt
                                  self.geoFence.xmax, #xo
                                  self.geoFence.ymax, #yo
                                  -0.5*self.omega2,
                                  -self.vel1
                                ])

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

        pos = pos[:2]
        offsetToTarget = offsetToTarget[:2]
        offsetToClosestObstacle = offsetToClosestObstacle[:2]
        # print("offsetToClosestObstacle", offsetToClosestObstacle)
        # print("offsetToTarget", offsetToTarget)
        # print("SHM velocity", self.obsInfo[2][1])
        # print("linear velocity", self.obsInfo[3][1])

        
        observation = np.concatenate([pos, offsetToTarget+pos, offsetToClosestObstacle+pos, [self.obsInfo[2][1][1], self.obsInfo[3][1][0]]])

        return observation
    
    def reset(self):
        self.episodeStepCount = 0
        self.trajectory = []
        self.noisyTrajectory = []
        self.dynamicObstaclesList = []
        self.simulationTime = 0
        self.obsInfo = {}
        self.obstacles = []
        self.offsetLine = None
        self.targetLine = None
        
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

        for Obstacle in self.dynamicObstaclesList:
                    if self.obsInfo[Obstacle][3] == 'shm':
                        self.moveObsSHM(Obstacle) 
                    if self.obsInfo[Obstacle][3] == 'linear':
                        self.moveObsLinear(Obstacle)
        self._updateobsInfo()

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
        y_initial = self.obsInfo[Obstacle][2][1]
        orientation = 0
        if orientation == 0:
            phi = asin(y_initial/0.5)
            velocity = self.omega2*0.5*cos(self.omega2*self.simulationTime+phi)
            p.resetBaseVelocity(Obstacle, [0, velocity, 0])
              
    def moveObsLinear(self, Obstacle):
        obsPos, obsOrn = p.getBasePositionAndOrientation(Obstacle)
        x,y,z = obsPos
        if x > 0.1:
            p.resetBaseVelocity(Obstacle, [-self.vel1, 0, 0])
        else:
            p.resetBaseVelocity(Obstacle, [0, 0, 0])
                 
    def step(self, action):
        self.VO_Reward = 0

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

        ################# FROM BASE CLASS #######################
            
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.SIM_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            self._saveLastAction(action)
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            self.simulationTime += 1/self.simFreq 

            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            # if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                # self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                assert self.PHYSICS == Physics.PYB , "Physics mode not implemented"
                self._physics(clipped_action[i, :], i)
                
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)

            ### Update Obs Dynamics ###
            if len(self.dynamicObstaclesList) > 0:
                for Obstacle in self.dynamicObstaclesList:
                    if self.obsInfo[Obstacle][3] == 'shm':
                        self.moveObsSHM(Obstacle)
                    if self.obsInfo[Obstacle][3] == 'linear':
                        self.moveObsLinear(Obstacle)
            self._updateobsInfo() #new velocity at new current postion arrived at based on earlier velocity

            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info
        
    def _computeReward(self):

        self.VO_Reward = 0
        state = self._getDroneStateVector(0)
        pos = state[:3]
        x,y = pos[0], pos[1]


        for obs in self.dynamicObstaclesList:
            velocity_vec_obs = self.obsInfo[obs][1]
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
        
        # if self.minObstacles >= 3:
        #     nObstacles = np.random.randint(2,6)
        # else:
        #     nObstacles = np.random.randint(self.minObstacles, self.maxObstacles)
        nObstacles = 2 #2 dynamic obstacles, if swapped with above if-else, first 2 obstacles will be dynamic
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

    def _updateobsInfo(self):
        for Obstacle in self.dynamicObstaclesList:
            pos, orient = p.getBasePositionAndOrientation(Obstacle)
            vel = p.getBaseVelocity(Obstacle)
            self.obsInfo[Obstacle][0] = pos[0:2]
            self.obsInfo[Obstacle][1] = vel[0][:2] #just x and y velocity
        # print("obsInfo", self.obsInfo)

    def _spawnObstacles(self):        
        moving_count = 0
        for obstaclePos in self.obstaclePositions:
            currObstacle = p.loadURDF('sphere_small.urdf', obstaclePos, globalScaling=2)
            p.changeDynamics(currObstacle, -1, mass=0)
            self.obstacles.append(currObstacle)
            if self.dynamicObstacles and moving_count<2:
                self.dynamicObstaclesList.append(currObstacle)
                pos, orient = p.getBasePositionAndOrientation(currObstacle)
                self.obsInfo[currObstacle] = [pos[:2], (0, 0), pos[:2], 'movement_type'] #[pos, velocity, initial_position, movement_type]
                if self.movementType == 1:
                    if currObstacle == 2: # '1' is probably the drone
                        self.obsInfo[currObstacle][3] = 'shm'
                    if currObstacle == 3:
                        self.obsInfo[currObstacle][3] = 'linear'
                moving_count+=1

    def _randomizeDroneSpawnLocation(self):
        y_scale = self.geoFence.ymax - self.geoFence.ymin
        self.initPos = np.array([self.geoFence.xmin + ObstacleAviary.MINOR_SAFETY_BOUND_RADIUS, 
                        (self.geoFence.ymin + self.geoFence.ymax) + np.random.uniform(-y_scale/2 + ObstacleAviary.COLLISION_BOUND_RADIUS*2, y_scale/2 - ObstacleAviary.COLLISION_BOUND_RADIUS*2),
                        (self.geoFence.zmin + self.geoFence.zmax)/2])

        self.INIT_XYZS = np.array([self.initPos])

