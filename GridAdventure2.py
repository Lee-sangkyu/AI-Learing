from typing import Self
import kymnasium as kym
import gymnasium as gym
import numpy as np
import pickle

N = 32 # ���� �� �� ����
AGENT_RIGHT, AGENT_DOWN, AGENT_LEFT, AGENT_UP = 1000, 1001, 1002, 1003 # ������Ʈ ������
GOAL, LAVA, WALL, = 810, 900, 250 # Ÿ�� �Ӽ�
RED_DOOR, GREEN_DOOR, BLUE_DOOR = 402, 412, 422 # ��� ������
RED_DOOR_OPEN, GREEN_DOOR_OPEN, BLUE_DOOR_OPEN = 400,410,420
RED_KEY,GREEN_KEY, BLUE_KEY = 500, 510, 520 #���� ����
RED, GREEN, BLUE = 0, 1, 2 #����
DIR ={0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # 움직임 좌표
ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_PICK, ACTION_DROP, ACTION_OPEN= 0, 1, 2 ,3,4,5
GAMMA, EVAL_THRESHOLD = 0.99, 1e-5 # ������ & ��å ���� �� ����

# ���� & �г�Ƽ
TURN_COST, MOVE_COST, WALL_PENALTY, PICK_PENALTY, DROP_PENALTY, OPEN_PENALTY = -0.5, -0.1, -10.0, -10.0, -100.0, -10.0
GOAL_REWARD, LAVA_REWARD, OPEN_REWARD, PICK_REWARD = 10000.0, -100.0, 10.0, 10.0

# (��, ��, ����)�� ���� ���� �ε����� ��ȯ
def encode_state(row, col, ori, key):
    return row * N * 16 + col * 16 + ori *4 + key

# ���� �ε����� (��, ��, ����)���� ��ȯ
def decode_state(state):
    return state // (N * 16), (state // 16) % N, (state//4) % 4, state % 4

# ���������� �÷��̾� ��ġ�� ���� ����
def parse_observation(obs):

    idx = np.where((obs == AGENT_RIGHT) | (obs == AGENT_DOWN) | (obs == AGENT_LEFT) | (obs == AGENT_UP))
    if len(idx[0]) > 0:
        pr, pc = int(idx[0][0]), int(idx[1][0])
        ori = {AGENT_UP: 0, AGENT_RIGHT: 1, AGENT_DOWN: 2, AGENT_LEFT: 3}[int(obs[pr, pc])]
    else:
        pr, pc, ori = 0, 0, 1
    idx_key = np.where((obs == RED_KEY) | (obs == GREEN_KEY) | (obs == BLUE_KEY))
    
    present = [
        np.any(obs == RED_KEY),    # 0
        np.any(obs == GREEN_KEY),  # 1
        np.any(obs == BLUE_KEY)    # 2
    ]

    # key 상태 결정
    if all(present):
        key = 3          # 3개 다 있음
    elif not present[0]:
        key = 0          # RED 없음
    elif not present[1]:
        key = 1          # GREEN 없음
    elif not present[2]:
        key = 2          # BLUE 없음
    else:
        key = 3          # 기본값

    return pr, pc, ori, key

class Agent(kym.Agent):
    def __init__(self, PI):
        self.PI = PI.astype(np.int8)
        self.next_action = None
        self.queued_action = None

    def save(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def act(self, observation, info):
        if self.queued_action is not None:
            action = self.queued_action
            self.queued_action = None
            return action

        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action

        pr, pc, ori,key = parse_observation(observation)
        action = int(self.PI[encode_state(pr, pc, ori, key)])

        if action == ACTION_RIGHT:
            idx_door=np.where(obs==GREEN_DOOR_OPEN)
            if len(idx_door[0])>0:
                dr, dc = DIR[ori]
                front_tile = observation[pr + dr, pc + dc]
                if front_tile == 100 and key == 1:  
                    self.queued_action = ACTION_DROP  # 회전 전에 드랍 수행
                    return self.queued_action

        if action == ACTION_LEFT:
            idx_door=np.where(obs==BLUE_DOOR_OPEN)
            if len(idx_door[0])>0:
                dr, dc = DIR[ori]
                front_tile = observation[pr + dr, pc + dc]
                if front_tile == 100 and key == 2:  
                    self.queued_action = ACTION_DROP # 회전 전에 드랍 수행
                    return self.queued_action

     
        # 만약 문을 열었으면 다음 행동을 forward로 예약
        if action == ACTION_OPEN:
            dr, dc = DIR[ori]
            front_tile = observation[pr + dr, pc + dc]
            if front_tile in [RED_DOOR, GREEN_DOOR, BLUE_DOOR]:  
                self.next_action = ACTION_FORWARD

        

        return action

if __name__ == "__main__":
    import os

    # pkl ������ �̹� ������ �ٷ� �׽�Ʈ
    if os.path.exists("agent2.pkl"):
        print("Load from agent2.pkl...")
        agent = Agent.load("agent2.pkl")

    # ������ �н� �� �׽�Ʈ
    else:
        print("Training new agent")
        # ȯ�� �ʱ�ȭ
        env = gym.make(id="kymnasium/GridAdventure-FullMaze-32x32-v0", render_mode="rgb_array", bgm=False)
        obs, _ = env.reset(seed=42)
        env_map = obs.copy()
        env.close()

        # ���� �� ����
        state_count = N * N * 4*4
        next_states = np.zeros((state_count, 6), dtype=np.int32)
        rewards = np.zeros((state_count, 6), dtype=np.float32)
        dones = np.zeros((state_count, 6), dtype=np.bool_)
        terminal_mask = np.zeros(state_count, dtype=np.bool_)
        door_table = np.zeros(3, dtype = np.bool_)
        door_table.fill(False)

        for state in range(state_count):
            row, col, ori, key = decode_state(state)
            
            tile = env_map[row, col]
            if tile in [WALL, GOAL, LAVA]:
                terminal_mask[state] = True
                next_states[state, :] = state
                dones[state, :] = True
                continue

            for action in range(6): 
                next_key = key
                if action == ACTION_LEFT: #0
                    next_row, next_col, next_ori = row, col, (ori - 1) % 4
                    reward, done = TURN_COST, False
                    
                elif action == ACTION_RIGHT: #1
                    next_row, next_col, next_ori = row, col, (ori + 1) % 4
                    reward, done = TURN_COST, False
                   
                elif action == ACTION_FORWARD: #2
                    dr, dc = DIR[ori]
                    next_row, next_col, next_ori = row + dr, col + dc, ori
                    if not (0 <= next_row < N and 0 <= next_col < N):
                        next_row, next_col, reward, done = row, col, WALL_PENALTY, False
                    else:
                        next_tile = env_map[next_row, next_col]
                        if next_tile in [WALL, RED_DOOR,GREEN_DOOR,BLUE_DOOR]:
                            next_row, next_col, reward, done = row, col, WALL_PENALTY, True
                        

                        elif next_tile == GOAL:
                            reward, done = GOAL_REWARD, True
                        elif next_tile == LAVA:
                            reward, done = LAVA_REWARD, True
                        else:
                            reward, done = MOVE_COST, False
                   
                elif action == ACTION_PICK: #3
                    if key !=3:
                        reward, done = PICK_PENALTY, True
                    else:
                        dr, dc = DIR[ori]
                        next_row, next_col, next_ori = row + dr, col + dc, ori
                        next_tile = env_map[next_row, next_col]
                        if next_tile == RED_KEY:
                            reward, done = PICK_REWARD, False
                            next_key = 0 
                        elif next_tile == GREEN_KEY:
                            reward, done = PICK_REWARD, False
                            next_key = 1
                        elif next_tile == BLUE_KEY:
                            reward, done = PICK_REWARD, False
                            next_key = 2 
                        else:
                            reward, done = PICK_PENALTY, True
                   
                elif action == ACTION_DROP: #4
                    if key == 3:
                        reward, done = DROP_PENALTY,True
                    else:
                        dr, dc = DIR[ori]
                        next_row, next_col, next_ori = row + dr, col + dc, ori
                        next_tile = env_map[next_row, next_col]
                        if next_tile == 100:
                            if (key==0 and door_table[RED]==True) or  (key==1 and door_table[GREEN]==True) or  (key==2 and door_table[BLUE]==True):
                                
                                reward, done = -1.0, False
                                next_key = 3
                            else: 
                                reward, done = DROP_PENALTY,True
                        
                        else:
                            reward, done = DROP_PENALTY, True

                    
                else :  #5
                    if key ==3:
                        reward, done = OPEN_PENALTY, True
                    else:
                        dr, dc = DIR[ori]
                        next_row, next_col, next_ori = row + dr, col + dc, ori
                        next_tile = env_map[next_row, next_col]
                        if next_tile == RED_DOOR:
                            if key == 0:
                                reward, done = OPEN_REWARD, False
                                door_table[RED]==True
                            
                                
                               
                            else:
                                reward, done = OPEN_PENALTY, True
                        elif next_tile == GREEN_DOOR:
                            if key== 1:
                                reward, done = OPEN_REWARD, False
                                door_table[GREEN]=True
                               
                              
                            else:
                                reward, done = OPEN_PENALTY, True
                        elif next_tile == BLUE_DOOR:
                            if key== 2 and door_table[BLUE]==False:
                                reward, done = OPEN_REWARD, False
                                door_table[BLUE]=True
                               
                               
                            else:
                                reward, done = OPEN_PENALTY, True
                      
                        else:
                            reward,done = OPEN_PENALTY, True
                    
                next_states[state, action] = encode_state(next_row, next_col, next_ori, next_key)
                rewards[state, action] = reward
                dones[state, action] = done
        state=next_states[state,action]
        
        

        # Policy Iteration
        policy = np.full(state_count, ACTION_RIGHT, dtype=np.int8)
        values = np.zeros(state_count, dtype=np.float32)

        for iteration in range(1, 101):
            # ��å ��
            for _ in range(1000):
                delta, new_values = 0.0, values.copy()
                for state in range(state_count):
                    if terminal_mask[state]:
                        new_values[state] = 0.0
                        continue
                    action = int(policy[state])
                    value = rewards[state, action] + (0 if dones[state, action] else GAMMA * values[next_states[state, action]])
                    delta = max(delta, abs(value - values[state]))
                    new_values[state] = value
                values = new_values
                if delta < EVAL_THRESHOLD:
                    break

            # ��å ����
            policy_stable = True
            for state in range(state_count):
                if terminal_mask[state]:
                    continue
                old_action = int(policy[state])
                best_action, best_value = old_action, -np.inf
                for action in range(6):
                    value = rewards[state, action] + (0 if dones[state, action] else GAMMA * values[next_states[state, action]])
                    if value > best_value:
                        best_value, best_action = value, action
                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False

            if policy_stable:
                break

        # ������Ʈ ����
        agent = Agent(PI=policy)
        agent.save("agent2.pkl")
        print("Agent saved to agent.pkl")

    # ������Ʈ �׽�Ʈ
    eval_env = gym.make(id="kymnasium/GridAdventure-FullMaze-32x32-v0", 
                        render_mode="human", 
                        bgm=False)
    obs, info = eval_env.reset(seed=42)
    done, step = False, 0

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        step += 1
        done = terminated or truncated
    

    print(f"Finish. {step} steps")

    
    
    
    
    
    
    
    
    
    
    
    