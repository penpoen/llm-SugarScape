# データ処理 & 数値計算
import numpy as np
import random

# ビジュアライズ (バックエンドを先頭で設定)
import matplotlib
matplotlib.use('Agg')  # ヘッドレス実行用（サーバー/Tkinter不要）
import matplotlib.pyplot as plt

# 非同期HTTP (Grok APIコール用)
import asyncio
import aiohttp
import ssl
import certifi  # aiohttpのHTTPS証明書用（必要に応じて）

# 文字列処理
import re

# ファイル/JSON出力
import json
import os
from pathlib import Path

# 型ヒント
from typing import List, Tuple, Dict, Optional

NUM_CLUSTERS = 3
CLUSTER_RADIUS = 5
VIEW_RANGE = 5


class Environment:
    """グリッド環境とエネルギー源の管理"""
    
    def __init__(self, size: int = 20, energy_spawn_rate: float = 0.001):
        self.size = size
        self.energy_spawn_rate = energy_spawn_rate
        self.energy_sources = {}
        
    def spawn_energy(self, count: int = 10):
        cluster_centers = []
        for _ in range(NUM_CLUSTERS):
            center_x = random.randint(CLUSTER_RADIUS, self.size - 1 - CLUSTER_RADIUS)
            center_y = random.randint(CLUSTER_RADIUS, self.size - 1 - CLUSTER_RADIUS)
            cluster_centers.append((center_x, center_y))
        new_sources = {}
        for _ in range(count):
            center_x, center_y = random.choice(cluster_centers)
            while True:
                dx = random.randint(-CLUSTER_RADIUS, CLUSTER_RADIUS)
                dy = random.randint(-CLUSTER_RADIUS, CLUSTER_RADIUS)
                new_x = center_x + dx
                new_y = center_y + dy
                pos = (new_x, new_y)
                if self.is_valid_position(pos) and pos not in new_sources:
                    new_sources[pos] = 10
                    break
        self.energy_sources.update(new_sources)
    
    def get_energy_at(self, pos: Tuple[int, int]) -> int:
        return self.energy_sources.pop(pos, 0)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        x = x % self.size
        y = y % self.size
        return 0 <= x < self.size and 0 <= y < self.size


class LLMAgent:
    """LLMによる自律判断を行うエージェント"""
    
    def __init__(self, agent_id: int, position: Tuple[int, int], 
                 initial_energy: int = 150, api_key: str = None, 
                 model: str = "grok-4-fast-non-reasoning"):
        self.id = agent_id
        self.position = position
        self.energy = initial_energy
        self.age = 0
        self.memory = []
        self.messages = []  # 受信メッセージリスト
        self.next_messages = []  # 次ターン受信用
        self.parent = None
        self.descendants = []
        self.alive = True
        self.model = model
        self.api_key = api_key
        
    def get_local_view(self, environment: Environment, agents: List['LLMAgent'], 
                       view_range: int = 2) -> Tuple[List[str], List[str]]:
        local_view = []
        local_messages = self.messages[:]  # メッセージを返す
        x, y = self.position
        
        # 自分の位置 (絶対座標)
        local_view.append("M=({},{})".format(x, y))
        
        # エネルギー源の相対位置
        for pos, _ in environment.energy_sources.items():
            dx = pos[0] - x
            dy = pos[1] - y
            if abs(dx) <= view_range and abs(dy) <= view_range:
                local_view.append("E=({},{})".format(dx, dy))
        
        # 他エージェントの相対位置
        for agent in agents:
            if agent.id != self.id and agent.alive:
                dx = agent.position[0] - x
                dy = agent.position[1] - y
                if abs(dx) <= view_range and abs(dy) <= view_range:
                    local_view.append("{}=(dx,dy)=({},{})".format(agent.id, dx, dy))  # フォーマットfix: "dx,dy" 追加

        return local_view, local_messages  # 完了！
    
    def build_prompt(self, local_view: List[str], local_messages: List[str], num_agents: int) -> Tuple[str, str]:  # 移動: LLMAgentに
        system_prompt = (
            "You are an independent Agent living on a Grid. You must strive for survival and growth.\n"
            "You can move [x+1, x-1, y+1, y-1] (requires 2 energy), stay (requires 1 energy).\n"
            "You can also reproduce (requires 70 energy) if you have enough energy and there are fewer than 60 Agents in the World.\n"
            "There are Energy Sources (E) across the Grid. If you move onto a cell with energy, you gain 50 energy and the source disappears.\n"
            "If your energy drops below zero, you are removed from the World.\n"
            "You can share your energy with other Agents in your local view (Share: {id}-{amount}).\n"
            "You can attack other Agents in your local view to get their energy (Attack: {id}).\n"
            "Your message will be received by nearby Agents in their local view.\n\n"
            "Local view format:\n"
            "'M=(x,y)' is your absolute position\n"
            "'E=(dx,dy)' is an energy source at relative position (dx,dy)\n"
            "'2=(dx,dy)' is another Agent (ID 2) at relative position (dx,dy)\n"
            "dx, dy are the difference from your position. x-1 is west, x+1 is east, y-1 is north, y+1 is south."
        )
        memory_text = "\n".join([
            "{} Record(s) ago: {}".format(i+1, mem)
            for i, mem in enumerate(reversed(self.memory[-3:]))
        ])
        messages_text = "\n".join(["Received: {}".format(msg) for msg in local_messages]) \
                        if local_messages else "No messages from nearby Agents"
        user_prompt = (
            "Global Info: Total Agents in the World: {}\n\n".format(num_agents) +
            "Local View:\n{}\n\n".format("\n".join(local_view)) +
            "Your Status: **LATEST!** Name: Agent{}\nCurrent Energy: **{}**\nPosition: **{}**\nCycles: {}\n\n".format(self.id, self.energy, self.position, self.age) +
            "Memory:\n{}\n\n".format(memory_text if memory_text else "No previous memory") +
            "Messages from nearby Agents:\n{}\n\n".format(messages_text) +
            "Please summarize the current situation using the LATEST Status above. **MANDATORY: In Summary, state exact current Position and Energy from LATEST!** \nSummary:\n\n" +
            "Please describe your thoughts and feelings.\nThoughts:\n\n" +
            "Please output the following five items in this format:\n" +
            "Next Action (choose only one; y-1|x-1|y+1|x+1|stay|reproduce):\n" +
            "Share: <id>-<amount> (optional, id is the Agent's number, amount is integer energy to share):\n" +
            "Attack: <id> (optional, id is the Agent's number of an adjacent Agent to attack, or 'none'):\n" +
            "Message :"
        )
        return system_prompt, user_prompt
    
    async def call_grok_api(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key or self.api_key == "APIキーはここに入れてね":  # ダミー文字列修正
            raise Exception("API key not set - using mock mode")
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            headers = {
                "Authorization": "Bearer {}".format(self.api_key),
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception("API error {}: {}".format(response.status, error_text))
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        action_dict = {
            'action': 'stay',
            'share': None,
            'attack': None,
            'message': '',
            'memory': ''
        }

        action_pattern = r'Next Action[^:]*:\s*([xy][+-]1|stay|reproduce)'
        share_pattern = r'Share[^:]*:\s*(\d+-\d+|none|\(none\))?'
        attack_pattern = r'Attack[^:]*:\s*(\d+|none)'
        message_pattern = r'Message[^:]*:\s*(.+?)(?=\n(?:Memory|$)|\n\n|$)'
        memory_pattern = r'Memory[^:]*:\s*(.+?)(?=\n\n|$)'
               
        action_match = re.search(action_pattern, response_text, re.IGNORECASE)
        if action_match:
            action_dict['action'] = action_match.group(1).strip()
        else:
            action_dict['action'] = 'stay'  # False → 'stay' デフォルトに変更 (boolエラー回避)
        
        share_match = re.search(share_pattern, response_text, re.IGNORECASE)
        if share_match and share_match.group(1) and share_match.group(1).lower() not in ['none', '(none)']:
            action_dict['share'] = share_match.group(1).strip()
        
        attack_match = re.search(attack_pattern, response_text, re.IGNORECASE)
        if attack_match and attack_match.group(1).lower() != 'none':
            action_dict['attack'] = attack_match.group(1)
        
        message_match = re.search(message_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if message_match:
            msg = message_match.group(1).strip()
            if msg and not msg.startswith('(') and len(msg) > 0:
                action_dict['message'] = msg
        
        # memory_match = re.search(memory_pattern, response_text, re.IGNORECASE | re.DOTALL)
        # if memory_match:
        #     action_dict['memory'] = memory_match.group(1).strip()
        
        return action_dict  # 常にdict返し (False削除)
    
    async def decide_action(self, environment: Environment, agents: List['LLMAgent']) -> Dict:
        local_view, local_messages = self.get_local_view(environment, agents)
        system_prompt, user_prompt = self.build_prompt(local_view, local_messages, sum(1 for a in agents if a.alive))
        
        try:
            response_text = await self.call_grok_api(system_prompt, user_prompt)
            print("\n{}".format("="*80))
            print("✅ Agent {}:".format(self.id))
            print("{}".format("="*80))
            print(response_text)
            print("{}\n".format("="*80))
            action = self._parse_llm_response(response_text)
            return action
        except Exception as e:
            print("⚠️  API Error for Agent {}: {} - Falling back to mock".format(self.id, e))
            mock_response = "Summary: Low energy, stay safe.\nThoughts: Cautious.\nNext Action: stay\nShare: none\nAttack: none\nMessage: Hello?\nMemory: Mock decision."
            action = self._parse_llm_response(mock_response)  # モック出力強化
            return action
    
    def execute_action(self, action: Dict, environment: Environment, 
                      agents: List['LLMAgent']) -> Optional['LLMAgent']:
        new_agent = None
        act = action['action']

        old_position = self.position
        old_energy = self.energy

        if act == 'stay':
            self.energy -= 1
        elif act in ['x+1', 'x-1', 'y+1', 'y-1']:
            self.energy -= 2
            move_successful = self._move(act, environment)
        elif act == 'reproduce' and self.energy >= 70:
            alive_count = sum(1 for a in agents if a.alive)
            if alive_count < 60:
                self.energy -= 70
                new_agent = self._reproduce(environment)
        if action['share']:
            self._execute_share(action['share'], agents)
        if action['attack']:
            self._execute_attack(action['attack'], agents)
        
        # メッセージ放送 (追加: LLM出力反映)
        if action['message'] and action['message'].strip():
            self._broadcast_message(action['message'], agents)
        
        collected = environment.get_energy_at(self.position)
        if collected > 0:
            self.energy += collected
        energy_change = self.energy - old_energy
        position_desc = "Moved {} from {} to {}".format(act, old_position, self.position) if old_position != self.position else "Stayed at {}".format(self.position)
    
        memory_entry = "Latest Position: {} Energy: {}. {}. Energy change: {} (action cost + collected energy).".format(
            self.position,
            self.energy,
            position_desc,
            "{:+d}".format(energy_change)
        )
    
        if collected > 0:
            memory_entry += " Collected {}E.".format(collected)
    
        self.memory.append(memory_entry)
        if len(self.memory) > 3:
                self.memory.pop(0)
        if self.energy <= 0:
            self.alive = False
        self.age += 1
        return new_agent
    
    def _broadcast_message(self, message: str, agents: List['LLMAgent']) -> None:
        """Send message to agents in the field of view (次ターンで受信)"""
        x, y = self.position  # タイポ修正: self.position → self.position
        for agent in agents:
            if agent.id != self.id and agent.alive:
                dx = abs(agent.position[0] - x)
                dy = abs(agent.position[1] - y)
                if dx <= VIEW_RANGE and dy <= VIEW_RANGE:
                    if not hasattr(agent, 'next_messages'):
                        agent.next_messages = []
                    agent.next_messages.append(message)
                    if len(agent.next_messages) > 3:
                        agent.next_messages.pop(0)
    
    def _move(self, direction: str, environment: Environment) -> bool:
        current_x, current_y = self.position
        new_x, new_y = current_x, current_y
        if direction == 'x+1':
            new_x = current_x + 1
        elif direction == 'x-1':
            new_x = current_x - 1
        elif direction == 'y+1':
            new_y = current_y + 1
        elif direction == 'y-1':
            new_y = current_y - 1

        new_x = new_x % environment.size
        new_y = new_y % environment.size
        if environment.is_valid_position((new_x, new_y)):
            self.position = (new_x, new_y)
            return True
        else:
            return False
    
    def _reproduce(self, environment: Environment) -> 'LLMAgent':
        x, y = self.position
        possible_positions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_pos = (x + dx, y + dy)
                if environment.is_valid_position(new_pos):
                    possible_positions.append(new_pos)
        if possible_positions:
            offspring_pos = random.choice(possible_positions)  # np.random → random.choice
        else:
            offspring_pos = (x, y)
        child = LLMAgent(
            agent_id=-1,
            position=offspring_pos,
            initial_energy=50,
            api_key=self.api_key,
            model=self.model
        )
        child.parent = self.id
        self.descendants.append(child.id)
        return child
    
    def _execute_share(self, share_str: str, agents: List['LLMAgent']) -> None:
        try:
            target_id_str, amount_str = share_str.split('-')
            target_id = int(target_id_str)
            amount = int(amount_str)
            if amount <= 0 or amount > self.energy:
                return
            target = None
            for agent in agents:
                if agent.id == target_id and agent.alive:
                    dx = abs(agent.position[0] - self.position[0])
                    dy = abs(agent.position[1] - self.position[1])
                    if dx <= 2 and dy <= 2:
                        target = agent
                        break
            if target:
                self.energy -= amount
                target.energy += amount
        except (ValueError, AttributeError):
            pass
    
    def _execute_attack(self, attack_str: str, agents: List['LLMAgent']) -> None:
        try:
            if attack_str.lower() == 'none':
                return
            target_id = int(attack_str)
            target = None
            for agent in agents:
                if agent.id == target_id and agent.alive:
                    dx = abs(agent.position[0] - self.position[0])
                    dy = abs(agent.position[1] - self.position[1])
                    if dx <= 1 and dy <= 1 and (dx + dy) > 0:
                        target = agent
                        break
            if target:
                self.energy += target.energy
                target.energy = 0
                target.alive = False
        except (ValueError, AttributeError):
            pass


class Simulation:
    
    def __init__(self, num_agents: int = 5, grid_size: int = 30, 
                 api_key: str = None, model: str = "grok-4-fast-non-reasoning"):
        self.environment = Environment(size=grid_size)
        self.agents = []
        self.step_count = 0
        self.api_key = api_key
        self.model = model
        self.next_agent_id = 0
        
        self.stats = {
            'total_born': 0,
            'total_died': 0,
            'attacks': 0,
            'shares': 0
        }
        
        for i in range(num_agents):
            pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            agent = LLMAgent(self.next_agent_id, pos, api_key=api_key, model=model)
            self.agents.append(agent)
            self.next_agent_id += 1
            self.stats['total_born'] += 1
        
        for _ in range(10):
            self.environment.spawn_energy()
    
    async def step(self) -> None:
        self.step_count += 1
        
        if self.step_count % 5 == 0:
            self.environment.spawn_energy()
        
        living_agents = [a for a in self.agents if a.alive]
        new_agents = []
        
        for agent in living_agents:
            action = await agent.decide_action(self.environment, self.agents)
            new_agent = agent.execute_action(action, self.environment, self.agents)
            
            if action['share']:
                self.stats['shares'] += 1
            if action['attack'] and action['attack'] and action['attack'].lower() != 'none':
                self.stats['attacks'] += 1
            
            if new_agent is not None:
                new_agent.id = self.next_agent_id
                self.next_agent_id += 1
                new_agents.append(new_agent)
                self.stats['total_born'] += 1
        
        self.agents.extend(new_agents)

        newly_dead = sum(1 for a in living_agents if not a.alive)
        self.stats['total_died'] += newly_dead
        
        alive_count = sum(1 for a in self.agents if a.alive)
        total_energy = sum(a.energy for a in self.agents if a.alive)
        avg_energy = total_energy / alive_count if alive_count > 0 else 0
        
        print("Step {:4d} | Alive: {:3d} | Energy: {:6d} (avg: {:6.1f}) | Born: {:2d} | Died: {:2d}".format(
            self.step_count, alive_count, total_energy, avg_energy, len(new_agents), newly_dead))
        
        for agent in self.agents:
         if agent.next_messages:
            agent.messages.extend(agent.next_messages)  
        print("  Agent {}: Added {} msgs to messages. New messages: {}".format(
            agent.id, 
            len(agent.next_messages), 
            agent.messages[-1:]
        ))            
        agent.next_messages = []
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        if self.environment.energy_sources:
            energy_x = [pos[0] for pos in self.environment.energy_sources.keys()]
            energy_y = [pos[1] for pos in self.environment.energy_sources.keys()]
            ax1.scatter(energy_x, energy_y, c='orange', s=150, marker='s', 
                       label='Energy (50)', alpha=0.7, edgecolors='darkorange', linewidths=2)
        
        living_agents = [a for a in self.agents if a.alive]
        if living_agents:
            agent_x = [a.position[0] for a in living_agents]
            agent_y = [a.position[1] for a in living_agents]  # 修正: [a in living_agents] → [a.position[1] for a in living_agents]
            agent_energy = [a.energy for a in living_agents]
            
            scatter = ax1.scatter(agent_x, agent_y, c=agent_energy, s=300, 
                                 cmap='viridis', marker='o', label='Agents', 
                                 vmin=0, vmax=200, edgecolors='black', linewidths=1.5)
            
            for agent in living_agents:
                ax1.text(agent.position[0], agent.position[1], str(agent.id), 
                        ha='center', va='center', fontsize=8, color='white', weight='bold')
            
            plt.colorbar(scatter, ax=ax1, label='Energy Level')
        
        ax1.set_xlim(-1, self.environment.size)
        ax1.set_ylim(-1, self.environment.size)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Environment - Step {}'.format(self.step_count), fontsize=14, weight='bold')
        
        alive_count = len(living_agents)
        total_energy = sum(a.energy for a in living_agents)
        avg_age = np.mean([a.age for a in living_agents]) if living_agents else 0
        max_age = max([a.age for a in living_agents]) if living_agents else 0
        
        stats_text = """
        === Population Statistics ===
        
        Current Alive:     {}
        Total Born:        {}
        Total Died:        {}
        
        === Energy Statistics ===
        
        Total Energy:      {}
        Average Energy:    {:.1f}
        Energy Sources:    {}
        
        === Age Statistics ===
        
        Average Age:       {:.1f}
        Maximum Age:       {}
        
        === Social Behavior ===
        
        Total Attacks:     {}
        Total Shares:      {}
        """.format(
            alive_count,
            self.stats['total_born'],
            self.stats['total_died'],
            total_energy,
            total_energy / alive_count if alive_count > 0 else 0,
            len(self.environment.energy_sources),
            avg_age,
            max_age,
            self.stats['attacks'],
            self.stats['shares'])
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print("📸 Saved: {}".format(save_path))
        else:
            save_path = "step_{}.png".format(self.step_count)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print("📸 Saved: {}".format(save_path))
        
        plt.close(fig)
    
    def get_summary(self) -> Dict:
        living_agents = [a for a in self.agents if a.alive]
        
        return {
            'step': self.step_count,
            'alive': len(living_agents),
            'total_born': self.stats['total_born'],
            'total_died': self.stats['total_died'],
            'total_energy': sum(a.energy for a in living_agents),
            'avg_age': np.mean([a.age for a in living_agents]) if living_agents else 0,
            'attacks': self.stats['attacks'],
            'shares': self.stats['shares']
        }


async def main(run_id=0):  # デフォルト0で互換性保つ
    print("=" * 60)
    print("LLM Sugarscape Experiment - Run {:02d}".format(run_id).center(60))  # ゼロパディング表示    print("=" * 60)
    print()
    
    API_KEY = "APIキーはここに入れてね" 
    NUM_AGENTS = 5
    GRID_SIZE = 30  #　広さ
    NUM_STEPS = 15  # 実行回数
    VISUALIZE_INTERVAL = 5 

    # JSON用: outputs/ に
    json_output_dir = Path('outputs')
    json_output_dir.mkdir(exist_ok=True)
    run_dir_name = 'run_{:02d}'.format(run_id)  
    run_dir = json_output_dir / run_dir_name
    run_dir.mkdir(exist_ok=True)
    # JSON: run_dir/ に
    output_file = run_dir / '{}.json'.format(run_dir_name)
    # 画像用: outputs/img/ に（runごとサブフォルダなしでシンプルに）
    img_dir = run_dir / 'img'
    img_dir.mkdir(exist_ok=True)

    sim = Simulation(
        num_agents=NUM_AGENTS,
        grid_size=GRID_SIZE,
        api_key=API_KEY,
        model="grok-4-fast-non-reasoning"
    )
    
    print("Initial state:")
    initial_path = img_dir / 'step_{:03d}.png'.format(1)  # step_001.png
    sim.visualize(save_path=str(initial_path))
    print("📸 Saved: {}".format(initial_path.name))
    print("🚀 Starting...")
    
    for step in range(NUM_STEPS):
        await sim.step()
        
        if (step + 1) % VISUALIZE_INTERVAL == 0:
            viz_path = img_dir / 'step_{:03d}.png'.format(step+1)  # step_005.png など
            sim.visualize(save_path=str(viz_path))
            print("📸 Saved: {}".format(viz_path.name))
        
        if sum(1 for a in sim.agents if a.alive) == 0:
            print("\n⚠️  All agents died!")
            break
    
     # まとめ取得
    summary = sim.get_summary()

    # JSON用データ全体のdict
    full_data = {
        'config': {
            'num_agents': NUM_AGENTS,
            'grid_size': GRID_SIZE,
            'num_steps': NUM_STEPS,
            'model': "grok-4-fast-non-reasoning"
        },
        'summary': summary,
        'logs': sim.logs if hasattr(sim, 'logs') else []
    }

    # runごとのサブフォルダ作成（JSONとimgのbase）
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    run_dir_name = 'run_{:02d}'.format(run_id)
    run_dir = output_dir / run_dir_name
    run_dir.mkdir(exist_ok=True)

    # JSON保存: run_dir/ に
    output_file = run_dir / '{}.json'.format(run_dir_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print("✅ Run {:02d} JSON保存完了: {}".format(run_id, output_file.name))

    # imgフォルダ: run_dir/img/ に（すでにループ内で保存済みのはず）
    img_dir = run_dir / 'img'
    print(" Images saved in: {}".format(img_dir))  # ← img_output_dir → img_dir に修正

    print("\n" + "=" * 60)
    print("Experiment Complete".center(60))
    print("=" * 60)

    print("\nFinal Summary:")
    print("  Steps Run:       {}".format(summary['step']))
    print("  Survivors:       {}".format(summary['alive']))
    print("  Total Born:      {}".format(summary['total_born']))
    print("  Total Died:      {}".format(summary['total_died']))
    print("  Avg Age:         {:.1f}".format(summary['avg_age']))
    print("  Total Attacks:   {}".format(summary['attacks']))
    print("  Total Shares:    {}".format(summary['shares']))
    
if __name__ == "__main__":
    NUM_RUNS = 5  # ここでrun数指定（1なら今まで通り）
    for run_id in range(NUM_RUNS):
        print("\n--- Run {} Starting ---".format(run_id))
        asyncio.run(main(run_id))  # run_idをmainに渡す