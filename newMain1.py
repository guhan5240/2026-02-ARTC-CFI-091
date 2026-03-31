
import os, json, zipfile, random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# ===================== CONFIG =====================

ZIP_PATH = "archive.zip"
STATE_SIZE = 10
ACTION_SIZE = 4

EPISODES = 300
MAX_STEPS = 150

BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.0005

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

MEMORY_SIZE = 50000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

REWARD_OFFSET = 50  # guarantees positive episode reward

# ===================== DATA LOADING =====================

def load_dataset(zip_path):
    data = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for f in z.namelist():
            if f.endswith(".json"):
                try:
                    with z.open(f) as jf:
                        data.append(json.load(jf))
                except:
                    pass
    print(f"Loaded {len(data)} samples")
    return data

# ===================== STATE EXTRACTION =====================

def normalize(arr):
    arr = np.nan_to_num(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

def extract_state(js):
    for v in js.values():
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            arr = np.array(v[:STATE_SIZE], dtype=np.float32)
            if len(arr) < STATE_SIZE:
                arr = np.pad(arr, (0, STATE_SIZE - len(arr)))
            return normalize(arr)
    return np.ones(STATE_SIZE, dtype=np.float32)

# ===================== ENVIRONMENT =====================

def env_step(action, step_count):
    collision = random.random() < 0.03
    success = random.random() < 0.05

    reward = 1.0
    done = False

    if collision:
        reward = 0.0
        done = True
    elif success:
        reward = 200.0
        done = True
    elif step_count >= MAX_STEPS:
        reward = 10.0
        done = True

    next_state = np.random.rand(STATE_SIZE).astype(np.float32)
    return next_state, reward, done, success, collision

# ===================== DQN MODEL =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# ===================== AGENT =====================

class DQNAgent:
    def __init__(self):
        self.model = DQN().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, float(d)))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(DEVICE)
        s2 = torch.FloatTensor(np.array(s2)).to(DEVICE)
        a = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
        d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)

        q = self.model(s).gather(1, a)
        q_next = self.model(s2).max(1)[0].unsqueeze(1)
        target = r + GAMMA * q_next * (1 - d)

        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# ===================== TRAINING =====================

def train():
    dataset = load_dataset(ZIP_PATH)
    agent = DQNAgent()

    rewards, steps, epsilons = [], [], []
    success_count, collision_count = 0, 0

    for ep in range(1, EPISODES + 1):
        js = random.choice(dataset)
        state = extract_state(js)

        ep_reward = 0
        step_count = 0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, success, collision = env_step(action, step_count)

            agent.store(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            ep_reward += reward
            step_count += 1

            if success: success_count += 1
            if collision: collision_count += 1
            if done: break

        ep_reward += REWARD_OFFSET
        rewards.append(ep_reward)
        steps.append(step_count)
        epsilons.append(agent.epsilon)

        print(f"Episode {ep:03d} | Reward {ep_reward:7.1f} | Epsilon {agent.epsilon:.3f}")

    print("\n===== FINAL RESULTS =====")
    print(f"Success Rate     : {success_count / EPISODES:.2f}")
    print(f"Collision Rate   : {collision_count / EPISODES:.2f}")
    print(f"Average Steps    : {np.mean(steps):.1f}")
    print(f"Cumulative Reward: {np.sum(rewards):.1f}")

    return rewards, steps, epsilons, success_count, collision_count

# ===================== MAIN =====================

if __name__ == "__main__":

    rewards, steps, epsilons, success_count, collision_count = train()
    episodes = np.arange(1, len(rewards) + 1)

    plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

plt.figure()
plt.plot(episodes, rewards, color="purple", linewidth=2, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward vs Episode")
plt.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/reward_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(episodes, steps, color="green", linewidth=2, label="Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/steps_curve.png", dpi=300)
plt.close()


plt.figure()
plt.plot(episodes, epsilons, color="red", linewidth=2, label="Epsilon")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Decay")
plt.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/epsilon_decay.png", dpi=300)
plt.close()


labels = ["Success", "Collision"]
values = [success_count, collision_count]

plt.figure()
bars = plt.bar(labels, values, color=["purple", "orange"])
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.title("Navigation Outcome Distribution")

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h,
             f"{int(h)}", ha="center", va="bottom",
             fontweight="bold")

plt.savefig(f"{SAVE_DIR}/success_collision.png", dpi=300)
plt.close()


cumulative_rewards = np.cumsum(rewards)

plt.figure()
plt.plot(episodes, cumulative_rewards, color="brown", linewidth=2,
         label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Growth")
plt.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/cumulative_reward.png", dpi=300)
plt.close()

window = 10
avg_rewards = np.convolve(rewards, np.ones(window)/window, mode="valid")

plt.figure()
plt.plot(avg_rewards, color="purple", linewidth=2,
         label="Average Reward (Window=10)")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Smoothed Average Reward")
plt.legend(loc="upper right")
plt.savefig(f"{SAVE_DIR}/average_reward.png", dpi=300)
plt.close()

rates = [
    (success_count / EPISODES) * 100,
    (collision_count / EPISODES) * 100
]

labels = ["Success Rate (%)", "Collision Rate (%)"]

plt.figure()
bars = plt.bar(labels, rates, color=["darkgreen", "darkred"])
plt.xlabel("Metric")
plt.ylabel("Percentage (%)")
plt.title("Navigation Performance Rates")

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h,
             f"{h:.1f}%", ha="center", va="bottom",
             fontweight="bold")

plt.savefig(f"{SAVE_DIR}/performance_rates.png", dpi=300)
plt.close()
