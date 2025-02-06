# DoomTaskResolver-Ver0.1(MSc Final Project)
###### 项目介绍

基于Vizdoom平台、强化学习和自然语言处理开发新的AI算法和环境生成算法。

使得AI代理能够通过新算法在算法随机生成的3D环境中达成NLP处理过的随机生成的任务并通过RL和DeepRL实现模型训练。

使用Python、pytorch、Vizdoom、Navdoom、Slade3、DeepRL-Grounding和一系列相关的lib实现。

由于代理在3D环境中训练消耗的时间过长，单GPU算力不够的关系，只实现了基本功能并没有对项目进行优化，导致毕设论文缺少了关键的evaluation对模型进行进一步的细致性能评估的部分。

基于A3C-LSTM算法更改的AI算法也没有在长时间步的情况下测试其中、长链思考的能力。


##### 该项目和拓展成期刊paper的写作已搁置，忙着找工作。




# 地图生成器/MAP GENERATOR


### 生成地图指示器和实际可用的wad地图
#### 请使用以下命令生成地图指示器：
```python
python maze.py {prefix}
```
这将会在generated map indicator文件夹下生成maze_'prefix'_MAP01至maze_'prefix'_MAP10共10张地图生成用的示意图。


#### 请使用以下命令生成训练与测试地图：
```python
python wad.py {prefix} {'your map`s name}
```
这将会调用generated map indicator文件夹中的示意图，并将其整合进同一个WAD中，生成一个 'your map's name'.wad 文件。

该WAD中包含了所有根据示意图生成的DOOM引擎可用的wad地图，共1-10关。

然后将wad地图手动移至DeepRL-Grounding-master文件夹的maps文件夹下。

# AI训练部分/AI Training Part
### 使用对应软件确保测试环境正常并使用对应指令开始正式训练过程
由于时间问题，对生成的wad地图文件采用外部Slade3软件打开的方式查看生成是否正常。
Slade3官网：https://slade.mancubus.net/index.php

#### 请使用以下命令开始训练过程：
```python
python env_test.py
```
env_test.py首先会检测DeepRL-Grounding-master文件夹中是否有上次训练所留存的模型，文件名为last_trained.pth。

如果有，则调用该预训练模型并继续后续训练。

如果没有，则将从0开始训练一个全新的模型。

训练完成后，env_test.py会将训练过的模型命名为last_trained.pth，并保存到DeepRL-Grounding-master文件夹内。


<details>
	<summary>更改constants.py中的数据可以更改训练难度。</summary>

```
 	SIZE_THRESHOLD:物品大小阈值。大于阈值则该物品为大，小于阈值则该物品为小。

	REWARD_THRESHOLD_DISTANCE：给于奖励的触发距离阈值。代理与物品间的距离小于这个距离，则认为代理获得了该物品，给于对应奖励。'Doom引擎中，通过检测玩家的hitbox与物品碰撞箱是否接触来判定是否捡起物品，可拾取物品的碰撞箱大小通常为20。'

	CORRECT_OBJECT_REWARD：奖励值。用于强化学习部分的奖惩机制实现。

	WRONG_OBJECT_REWARD：惩罚值。用于强化学习部分的奖惩机制实现。

 	MAP_SIZE_X, MAP_SIZE_Y:地图XY轴大小。用于地图生成。过大的地图会导致地形变复杂，代理在探索过程中的寻路时间会变长。

 其余变量为原项目中的变量，训练过程中并未用到，因此保留并未更改。
```
</details>




---
<details>
 <summary>自省与废话</summary>

  ###### 没能彻底完成该项目也算是一年硕的遗憾，但至少学到了非常多的相关知识，值得。

  ###### 正在思考通过将Socratic Learning这种方法和LLM相结合来实现让AI拥有真正的自我思考的能力。但是Scoratic Learning也有其局限性，算力消耗会进一步提高。

  ###### 开源是当初和导师共同讨论的事，我个人一直都认为开源是计算机创新进步的动力之一，人人敝帚自珍反倒是一种阻力。
</details>
