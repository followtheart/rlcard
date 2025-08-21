# RLCard 源码分析与二次开发教材

## 简介

本教材提供了 RLCard 项目的全面源码分析和二次开发指南，旨在帮助开发者深入理解 RLCard 的架构设计，并掌握扩展开发的方法。

## 教材内容

### 📖 [主要教材文档](source-code-analysis-and-development-guide.md)

详细的教材包含以下章节：

1. **项目概述与架构分析** - 了解 RLCard 的整体设计理念
2. **源码结构详细分析** - 深入理解代码组织结构  
3. **核心组件深度剖析** - 掌握环境、游戏、智能体三大核心系统
4. **二次开发指南** - 学习如何添加新游戏和新算法
5. **实战案例与示例** - 通过具体例子学习开发技巧
6. **开发工作流程** - 掌握从开发到部署的完整流程
7. **最佳实践与规范** - 遵循高质量代码标准

### 🚀 [实战示例代码](../examples/tutorial_demo.py)

配套的实践代码，演示：
- 环境状态分析
- 智能体实现模式
- 性能评估方法
- 自定义配置示例

## 快速开始

### 运行教材示例

```bash
# 基础环境分析
python examples/tutorial_demo.py --env blackjack --mode analyze

# 智能体性能比较
python examples/tutorial_demo.py --env blackjack --mode compare

# 环境自定义演示
python examples/tutorial_demo.py --env blackjack --mode customize

# 运行所有示例
python examples/tutorial_demo.py --env blackjack --mode all
```

### 支持的环境

- `blackjack` - 二十一点（推荐初学者）
- `leduc-holdem` - 简化德州扑克
- `limit-holdem` - 限注德州扑克
- `doudizhu` - 斗地主

### 示例输出

```
环境分析: blackjack
玩家数量: 1
动作空间大小: 2
状态空间形状: [[2]]

=== 步骤 1 状态分析 ===
观察向量: [11 11]
合法动作: [0, 1]
原始合法动作: ['hit', 'stand']
```

## 学习路径

### 初学者路径 🌱
1. 阅读教材第1-2章，了解整体架构
2. 运行示例代码，观察环境行为
3. 学习第3章，理解核心组件
4. 尝试修改示例代码

### 进阶开发者路径 🔥
1. 深入学习第4章二次开发指南
2. 参考第5章实战案例
3. 按照第6章建立开发环境
4. 遵循第7章最佳实践

### 高级贡献者路径 ⚡
1. 完整掌握所有章节内容
2. 参与开源贡献
3. 开发新游戏环境
4. 实现创新算法

## 相关资源

- [RLCard 官方文档](http://www.rlcard.org)
- [RLCard GitHub 仓库](https://github.com/datamllab/rlcard)
- [RLCard 教程](https://github.com/datamllab/rlcard-tutorial)
- [贡献指南](../CONTRIBUTING.md)

## 反馈与贡献

欢迎通过以下方式参与改进：

- 🐛 [报告问题](https://github.com/datamllab/rlcard/issues)
- 💡 提出改进建议
- 📝 完善文档内容
- 🔧 提交代码修改

## 许可证

本教材遵循 MIT 许可证，与 RLCard 项目保持一致。

---

**开始学习：** [点击这里查看完整教材](source-code-analysis-and-development-guide.md)