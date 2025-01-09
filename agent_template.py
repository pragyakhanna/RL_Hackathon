# agent_template.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            # Increased Layer size
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class LunarLanderAgent:
    def __init__(self):
        # Initialize environment
        self.env = gym.make('LunarLander-v3')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN hyperparameters
        self.state_size = 8
        self.action_size = 4
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 1e-5
        self.batch_size = 64
        self.memory = deque(maxlen=100000)
        
        # Initialize networks
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Compute next Q values
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def train(self, num_episodes):
        best_avg_reward = float('-inf')
        scores = deque(maxlen=100)
        avg_rewards = []
        plot_points = [250,500,750,1000]
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            scores.append(total_reward)
            avg_reward = np.mean(scores)
            avg_rewards.append(avg_reward)
            
            if len(scores) >= 100 and avg_reward > best_avg_reward and self.epsilon <= 0.1:
                best_avg_reward = avg_reward
                self.save_model('model.pkl')
            
            if episode % 10 == 0:
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if(episode+1) in plot_points:
                plt.figure()
                plt.plot(range(1, episode + 2), avg_rewards, label='Average Reward')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.title(f'Learning Curve')
                plt.legend()
                plt.show()

                

    def test(self, num_episodes=100):
        scores = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
            
            scores.append(total_reward)
        
        avg_score = np.mean(scores)
        print(f"Average Test Score: {avg_score:.2f}")
        return avg_score

    def save_model(self, file_name):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_name)

    def load_model(self, file_name):
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {file_name}")
        else:
            print(f"No model file found at {file_name}")

if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'model.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model

    num_training_episodes = 1000  # Define the number of training episodes
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    agent.save_model(agent_model_file)
    print("Model saved.")
