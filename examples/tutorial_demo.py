#!/usr/bin/env python3
"""
RLCard 源码分析与二次开发教材 - 实战示例

这个脚本演示了如何:
1. 分析RLCard环境的基本结构
2. 理解状态表示和动作编码
3. 实现简单的智能体
4. 评估算法性能

作者：RLCard 开发团队
"""

import rlcard
import numpy as np
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
import argparse


class SimpleAnalysisAgent:
    """
    简单分析智能体 - 展示如何理解环境状态
    这个智能体会分析游戏状态并打印详细信息
    """
    
    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions
        self.step_count = 0
        
    def step(self, state):
        """训练步骤 - 分析状态并选择动作"""
        self.step_count += 1
        self._analyze_state(state)
        
        # 选择随机合法动作
        legal_actions = list(state['legal_actions'].keys())
        action = np.random.choice(legal_actions)
        
        print(f"  → 选择动作: {action} ({state['raw_legal_actions'][legal_actions.index(action)]})")
        return action
    
    def eval_step(self, state):
        """评估步骤"""
        action = self.step(state)
        
        # 计算动作概率
        probs = [0 for _ in range(self.num_actions)]
        legal_actions = list(state['legal_actions'].keys())
        for i in legal_actions:
            probs[i] = 1.0 / len(legal_actions)
            
        info = {'probs': probs}
        return action, info
    
    def _analyze_state(self, state):
        """分析状态信息"""
        print(f"\n=== 步骤 {self.step_count} 状态分析 ===")
        print(f"观察向量形状: {state['obs'].shape}")
        print(f"观察向量: {state['obs']}")
        print(f"合法动作: {list(state['legal_actions'].keys())}")
        print(f"原始合法动作: {state['raw_legal_actions']}")
        
        if 'raw_obs' in state:
            print("原始观察信息:")
            for key, value in state['raw_obs'].items():
                print(f"  {key}: {value}")


def analyze_environment(env_name, num_games=3):
    """分析环境的基本信息"""
    print(f"\n{'='*50}")
    print(f"环境分析: {env_name}")
    print(f"{'='*50}")
    
    # 创建环境
    env = rlcard.make(env_name, config={'seed': 42})
    
    # 打印环境基本信息
    print(f"玩家数量: {env.num_players}")
    print(f"动作空间大小: {env.num_actions}")
    print(f"状态空间形状: {env.state_shape}")
    print(f"动作空间形状: {env.action_shape}")
    
    # 创建分析智能体
    agents = []
    for i in range(env.num_players):
        if i == 0:
            # 第一个玩家使用分析智能体
            agent = SimpleAnalysisAgent(env.num_actions)
        else:
            # 其他玩家使用随机智能体
            agent = RandomAgent(env.num_actions)
        agents.append(agent)
    
    env.set_agents(agents)
    
    # 运行多个游戏进行分析
    total_payoffs = []
    for game_idx in range(num_games):
        print(f"\n--- 游戏 {game_idx + 1} ---")
        
        trajectories, payoffs = env.run(is_training=False)
        total_payoffs.append(payoffs)
        
        print(f"游戏结果: {payoffs}")
        print(f"轨迹长度: {[len(traj) for traj in trajectories]}")
    
    # 分析结果
    print(f"\n{'='*30}")
    print("性能分析:")
    print(f"{'='*30}")
    
    avg_payoffs = np.mean(total_payoffs, axis=0)
    std_payoffs = np.std(total_payoffs, axis=0)
    
    for i, (avg, std) in enumerate(zip(avg_payoffs, std_payoffs)):
        print(f"玩家 {i}: 平均收益 = {avg:.3f} ± {std:.3f}")


def compare_agents(env_name, num_games=100):
    """比较不同智能体的性能"""
    print(f"\n{'='*50}")
    print(f"智能体性能比较: {env_name}")
    print(f"{'='*50}")
    
    env = rlcard.make(env_name, config={'seed': 42})
    
    # 随机智能体
    random_agent = RandomAgent(env.num_actions)
    env.set_agents([random_agent] * env.num_players)
    
    random_payoffs = []
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        random_payoffs.append(payoffs[0])  # 只看第一个玩家
    
    avg_random = np.mean(random_payoffs)
    std_random = np.std(random_payoffs)
    
    print(f"随机智能体: {avg_random:.3f} ± {std_random:.3f}")
    
    # 分析智能体（实际上也是随机的，但有分析功能）
    analysis_agent = SimpleAnalysisAgent(env.num_actions)
    other_agents = [RandomAgent(env.num_actions) for _ in range(env.num_players - 1)]
    env.set_agents([analysis_agent] + other_agents)
    
    analysis_payoffs = []
    for _ in range(min(5, num_games)):  # 少运行几次，因为有分析输出
        _, payoffs = env.run(is_training=False)
        analysis_payoffs.append(payoffs[0])
    
    avg_analysis = np.mean(analysis_payoffs)
    std_analysis = np.std(analysis_payoffs)
    
    print(f"分析智能体: {avg_analysis:.3f} ± {std_analysis:.3f}")


def demonstrate_customization():
    """演示环境自定义"""
    print(f"\n{'='*50}")
    print("环境自定义演示")
    print(f"{'='*50}")
    
    # 展示不同配置的环境创建
    configs = [
        {'seed': 42},
        {'seed': 123, 'allow_step_back': True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i + 1}: {config}")
        env = rlcard.make('blackjack', config=config)
        
        agent = RandomAgent(env.num_actions)
        env.set_agents([agent] * env.num_players)
        
        _, payoffs = env.run(is_training=False)
        print(f"结果: {payoffs}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RLCard 源码分析与开发教材示例")
    parser.add_argument(
        '--env', 
        type=str, 
        default='blackjack',
        choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu'],
        help='选择要分析的环境'
    )
    parser.add_argument(
        '--games', 
        type=int, 
        default=3,
        help='分析的游戏数量'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='analyze',
        choices=['analyze', 'compare', 'customize', 'all'],
        help='运行模式'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子以保证可重现性
    set_seed(42)
    
    print("RLCard 源码分析与二次开发教材 - 实战示例")
    print("本示例演示了环境分析、智能体实现和性能评估的基本方法")
    
    if args.mode == 'analyze' or args.mode == 'all':
        analyze_environment(args.env, args.games)
    
    if args.mode == 'compare' or args.mode == 'all':
        compare_agents(args.env, min(args.games * 10, 100))
    
    if args.mode == 'customize' or args.mode == 'all':
        demonstrate_customization()
    
    print(f"\n{'='*50}")
    print("示例完成！")
    print("更多详细信息请参考 docs/source-code-analysis-and-development-guide.md")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()