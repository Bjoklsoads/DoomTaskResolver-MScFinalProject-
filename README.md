# DoomTaskResolver
硕士学位毕设项目。
基于python3.13和一系列相关的lib。

由于代理在3D环境中训练消耗的时间过长的关系，只实现了基本功能并没有对项目进行优化和对模型进行进一步的细致性能评估，导致毕设论文缺少了关键的evaluation部分。

基于A3C-LSTM更改的AI算法也没有在长时间步的情况下测试其中、长链思考的能力。
<details>
  <summary></summary>
  
  算是一年硕的一种遗憾。
  
</details>


##### 该项目暂时搁置，忙着找工作和申请博士研究。
##### 正在思考通过将Socratic Learning这种方法和LLM相结合来实现让AI拥有真正的，自我思考、自我递归的能力。但是Scoratic Learning也有其局限性，算力消耗会进一步提高。




# MAP GENERATOR

## 生成地图指示器和实际可用的wad地图
请使用以下命令生成地图指示器：
```python
python maze.py {'your map indicator`s prefix name'}
```
这将会在generated map indicator文件夹下生成maze_'prefix'_MAP01至maze_'prefix'_MAP10共10张地图生成用的示意图。

生成训练与测试地图使用命令
```python
python wad.py {'your map indicator`s prefix name'}
```
这将会调用generated map indicator文件中的示意图，并将其整合进同一个WAD中，生成一个room.wad文件
