import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
from tqdm import tqdm_notebook
import numpy as np
from collections import deque
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns


# Q function 用于计算动作价值函数，给定状态和动作，将会生成对应的Q值
class QFunction():
    
    # 初始化获取权重向量
    def __init__(self, weight_vector):
        self.weight_Q = weight_vector
    
    def __call__(self, state, action):
        
        # Policy（state）输入一个状态，将会得到确定性策略下对应的动作
        a_pred = Policy(state)
        
        # 计算输入的动作与策略动作的差值
        error = action - a_pred
        
        # Policy.weight_policy表示策略Policy的权重参数
        Policy.weight_policy.grad = None
        # backward()是 PyTorch 中自动求导的函数，用于计算损失函数关于模型参数的梯度
        a_pred.backward()
        
        # Vector_sa是一个状态-动作特征向量，由参数weight_policy的梯度和时间差分错误相乘得到
        Vector_sa = Policy.weight_policy.grad * error
        
        # 状态-动作特征向量与权重向量相乘，计算采取给定动作而非当前策略规定动作的优势
        Advantage_sa = multipleVector(Vector_sa, self.weight_Q)
        
        # 状态价值函数 (V(state)) 表示在给定状态下执行策略的期望回报
        # 这里加上上一步计算出的优势值得到执行输入动作的状态-动作值
        q_value = Advantage_sa + V(state)
        
        return q_value

# DeterministicPolicy 用于计算确定性策略，给定状态，将会生成对应的动作
class DeterministicPolicy():
    
    # 初始化获取权重向量
    def __init__(self, weight_vector):
        self.weight_policy = weight_vector
    
    def __call__(self, state):
        
        # 计算权重向量和状态向量相乘的结果
        action = multipleVector(self.weight_policy, state)
        
        # 使用函数 torch.tanh 进行变换，使得动作的范围限制在 [-1, 1] 之间
        action = torch.tanh(action)
        
        return action

# ValueFunction 用于计算状态价值函数，给定状态，将会生成对应的价值
class ValueFunction():
    
    def __init__(self, weight_vector):
        self.weight_value = weight_vector
    
    def __call__(self, state):
        
        # 计算权重向量和状态向量相乘的结果
        val = multipleVector(self.weight_value, state)
        
        return val


# 计算权重向量和特征向量的乘积
def multipleVector(weight_vector, feature_vector):
    # 输入为 权重向量和特征向量

    # 转置权重向量并乘以特征向量
    product = torch.matmul(torch.transpose(weight_vector, 0, 1), feature_vector)
    
    return product


def normalizeState(state, mean, std):
    # 对状态进行归一化处理，处理方式为：（状态-均值）/方差
    # 归一化的目的是将数据调整到同一范围内，以便模型能够更好地处理这些数据

    # state是当前状态，mean是归一化的均值，std是归一化的方差
    # mean和std是在训练过程中计算出来的

    normalized = (state - mean) / std
    
    return normalized

def update(batch, state_mean, state_std):
    # 使用单批次样本数据更新参数
    # batch是样本数据，state_mean是归一化的均值，state_std是归一化的方差    

    for sample in batch:
        
        # 分解样本值
        state, action, new_state, reward, done = sample

        # 将状态进行归一化处理
        state = normalizeState(state, state_mean, state_std)
        new_state = normalizeState(new_state, state_mean, state_std)

        # 将状态转化为张量，其中unsqueeze(1）在张量的第一维插入一个新的维度
        state_tensor = torch.from_numpy(state).float().unsqueeze(1)  
        new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(1)

        # 计算当前状态下的动作
        new_action = Policy(new_state_tensor)

        # 计算时间差分误差
        error = reward + discountFactor * Q(new_state_tensor, new_action) - Q(state_tensor, action)

        # 使用自然梯度更新weight_policy，其中weight_policy是策略的参数
        Policy.weight_policy.grad = None
        Policy(state_tensor).backward()

        # 计算雅可比矩阵
        jacob_matrix = Policy.weight_policy.grad

        #更新参数weight_policy，weight_policy_update = jacob_matrix * multipleVector(jacob_matrix, Q.weight_Q) 
        weight_policy_update = Q.weight_Q

        # Vector_sa 表示状态-动作特征向量， error 表示时间差分误差
        # 更新参数weight_Q，weight_Q_update = error * Vector_sa
        Vector_sa = (action - Policy(state_tensor)) * jacob_matrix
        weight_Q_update = error.detach() * Vector_sa.detach()

        # 更新参数weight_value，weight_value_update = error * jacob_matrix
        weight_value_update = error * jacob_matrix

        # 更新参数weight_policy，weight_Q，weight_value
        Policy.weight_policy = Policy.weight_policy.detach() + learningRate_policy * weight_policy_update # 分离计算图，防止梯度传播
        Policy.weight_policy.requires_grad = True # 上一行的计算进行了分离，因此需要重新设置为需要梯度计算

        Q.weight_Q = Q.weight_Q.detach() + learningRate_Q * weight_Q_update
        V.weight_value = V.weight_value.detach() + learningRate_value * weight_value_update

def draw_result(scores, policy_scores):
    plt.plot(scores, color='green', label='Training score')
    plt.plot(policy_scores, color='red', label='Policy score')
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('COPDAC-Q')
    plt.legend(prop={'size': 12},bbox_to_anchor=(1, 0.6))
    plt.show()


def train():
    torch.set_printoptions(precision=10) # 设置浮点数的精度
    discountFactor = 0.99 # 设置折扣因子
    NUM_EPISODES = 100 # 设置循环次数
    MAX_STEPS = 5000 # 设置每次最大步数
    SOLVED_SCORE = 90 # 解决问题的最低分数
    learningRate_policy = 0.005 # 策略的学习率
    learningRate_value = 0.03 # 状态价值函数的学习率
    learningRate_Q = 0.03 # 动作价值函数的学习率
    BATCH_SIZE = 8 # 每次训练的批次大小
    # 构建环境
    env = gym.make('MountainCarContinuous-v0').env
    env2 = deepcopy(env)
    # 设置环境参数
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    # 设置随机种子
    np.random.seed(0)
    random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    # 初始化权重向量
    stdv = 1 / np.sqrt(obs_space)
    weight_policy = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, action_space)) * 0.03)
    weight_policy.requires_grad = True
    weight_Q = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, 1)) * 0.03)
    weight_value = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, 1)) * 0.03)
    # 初始化策略，状态价值函数，动作价值函数
    Policy = DeterministicPolicy(weight_policy)
    Q = QFunction(weight_Q)
    V = ValueFunction(weight_value)
    # 初始化经验池
    samples = np.array([env.observation_space.sample() for _ in range(10000)])
    state_mean = np.mean(samples, axis = 0)
    state_std = np.std(samples, axis= 0) + 1.0e-6
    # 记录每次训练的分数
    scores = []
    # 记录每次训练的策略分数
    policy_scores = []
    # 统计更新次数
    update_count = 0
    # 经验回放的缓冲区：大小为8000的双端队列
    replay_buffer = deque(maxlen=8000)
    
    for episode in tqdm_notebook(range(NUM_EPISODES)):
        
        state = env.reset()
        state2 = env2.reset()
        
        score = 0
        score2 = 0
        
        done = False
        done2 = False
        
        for step in range(MAX_STEPS):  
            # 在动作空间中随机采样
            action = env.action_space.sample()[0]
            
            # 执行上一步得到的动作，并返回新的状态、奖励和终止信息
            new_state, reward, done, _ = env.step([action])

            # 累加分数
            score += reward
            # 计算机械能的变化量，并将其×100后累加计入奖励值
            reward += 100*((np.sin(3*new_state[0]) * 0.0025 + 0.5 * new_state[1] * new_state[1]) - (np.sin(3*state[0]) * 0.0025 + 0.5 * state[1] * state[1]))
            
            # 将当前状态、动作、新状态、奖励和终止信息存入经验池
            item = [state, action, new_state, reward, done]
            replay_buffer.append(item)
            
            # 每10次训练更新一次策略
            # if step % 10 == 0 and len(replay_buffer) > BATCH_SIZE:
            if step % BATCH_SIZE == 0 and len(replay_buffer) >= BATCH_SIZE:
                replay = random.sample(replay_buffer, BATCH_SIZE) # 从缓存中取出BATCH_SIZE个样本
                update(replay, state_mean, state_std) # 更新参数
                update_count += 1
        
            if done:
                break
            
            state = new_state
        
        # 记录每次训练的分数
        scores.append(score)        
        # 测试当前得到的策略
        for step in range(MAX_STEPS):
            state2 = normalizeState(state2, state_mean, state_std) # 状态归一化
            state_tensor2 = torch.from_numpy(state2).float().unsqueeze(1) # 将状态转换为张量
            action2 = Policy(state_tensor2) # 获取策略中状态对应的动作
            new_state2, reward2, done2, _ = env2.step([action2.item()]) # 执行动作
            score2 += reward2 # 累加分数            
            if done2:
                break                
            state2 = new_state2
        
    testing_scores = []
    for _ in tqdm_notebook(range(100)):
        state = env.reset()
        done = False
        score = 0
        for step in range(MAX_STEPS):
            #env.render()
            state = normalizeState(state, state_mean, state_std)
            state_tensor = torch.from_numpy(state).float().unsqueeze(1)
            action = Policy(state_tensor)
            new_state, reward, done, info = env.step([action.sample()])            
            score += reward            
            state = new_state            
            if done:
                break
        testing_scores.append(score)
    env.close()
    print('The mean score of testing is:')
    print(np.array(testing_scores).mean())
    print('The variance of testing is:')
    print(np.array(testing_scores).var())            
    policy_scores.append(score2) # 记录每次训练的策略分数
    draw_result(scores, policy_scores) # 绘制训练过程中的分数变化图
    

if __name__ == '__main__':
    train()
