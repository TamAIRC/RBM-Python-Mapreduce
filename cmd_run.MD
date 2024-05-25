**Active Deactivate .venv**

```bash
.\venv\Scripts\activate.bat
```

```bash
deactivate
```

**Remove folder/file**
```bash
hdfs dfs -rm -r /path/to
```

**Format namenode**

```bash
hdfs namenode -format
```

**Format namenode**

```bash
hdfs datanode -format
```
**Put file to hdfs**
```bash
hdfs dfs -put /local/path/to/file /path/to/file
```
**Fix bug datanode shutdown**
- Windows
Open and Get clusterID: "C:\hadoop-3.3.0\data\namenode\current\VERSION"
Open and Paste clusterID: "C:\hadoop-3.3.0\data\datanode\current\VERSION"


## Train

```bash
python ./hadoop_rbm/mr_deep_learning_job.py -r hadoop ./hadoop_rbm/input_mnist/* --model-path ./hadoop_rbm/model/rbm.pth
```

```bash
python .\hadoop_rbm\mr_deep_learning_job.py -r hadoop hdfs:///input_mnist/* --model-path hdfs:///model/rbm.pth --hadoop-streaming-jar C:\hadoop-3.3.0\share\hadoop\tools\lib\hadoop-streaming-3.3.0.jar
```
```bash
python .\hadoop_rbm\mr_simple_job.py -r hadoop hdfs:///input_mnist/train --hadoop-streaming-jar C:\hadoop-3.3.0\share\hadoop\tools\lib\hadoop-streaming-3.3.0.jar
```


```bash
python ./mr_deep_learning_job.py -r hadoop ./input_mnist/train --model-path ./model/rbm.pth --hadoop-streaming-jar C:/hadoop-3.3.0/share/hadoop/tools/lib/hadoop-streaming-3.3.0.jar
```