# BigDataHomeworks
## Homeworks for the course on Big Data Computing, held @University of Padua

## Homework 4 

### Accesing the cluster:

putty: `gajdusek@login.dei.unipd.it`

when the connection opens: `ssh -p 2222 group50@147.162.226.106` (insert the new password...)

to upload the file from windows to machine in the lab
- check that you have putty folder in the PATH environmental variable 
- in windows cmd: `pscp C:\Users\pavel\PycharmProjects\BigDataAll\HW3\HW3_files\G50HM4.py gajdusek@login.dei.unipd.it:Downloads`

to upload the file from machine in the lab to the cluster (in the putty command line)
- `scp -P 2222 G50HM4.py group50@147.162.226.106:G50HM4.py`
- log into the cluster: `ssh -p 2222 group50@147.162.226.106`
- `hdfs dfs -ls /tmp`
- copy the file from `/tmp` to our folder: `hdfs dfs -copyFromLocal /tmp/G50HM4.py G50HM4.py`
- check that the file was copied: `hdfs dfs -ls`

- `yarn application -list`


****TODO****: running jobs:

TODO: I don't know why, but I don't manage to run the script from our local folder...
It shows that the file was not found. So I run it from tmp. 

According to the webpage of hw:

- `spark-submit --conf spark.pyspark.python=python3 --num-executors 4 G50HM4.py /data/ `
- maximum X: 32
- to pass one of preloaded files as an argument: specify path `/data/filename`, e.g. `/data/HIGGS11M7D.txt.gz`

for more details see [here](http://www.dei.unipd.it/~capri/BDC/HOMEWORKS/homework4.html)
