# BigDataHomoworks
## Homeworks for the course on Big Data Computing, held @University of Padua

## Homework 4 

### Accesing the cluster:

putty: `gajdusek@login.dei.unipd.it`

when the connection opens: `ssh -p 2222 group50@147.162.226.106` (insert the new password...)

to upload the file from windows to machine in the lab
- check that you have putty folder in the PATH environmental variable 
- `pscp C:\Users\path\to\your\folder\G50HM3.py gajdusek@login.dei.unipd.it:Downloads`

to upload the file from machine in the lab to the cluster
- `scp -P 2222 G50HM3.py group50@147.162.226.106:/tmp`
- log into the cluster: `ssh -p 2222 group50@147.162.226.106`
- `hdfs dfs -ls /tmp`
- copy the file from `/tmp` to our folder: `hdfs dfs -copyFromLocal /tmp/G50HM3.py G50HM3.py`
- check that the file was copied: `hdfs dfs -ls`

****TODO****: running jobs:

According to the webpage of hw:

- `spark-submit --conf spark.pyspark.python=python3 --num-executors X GxxHM4.py argument-list `
- maximum X: 32
- to pass one of preloaded files as an argument: specify path `/data/filename`, e.g. `/data/HIGGS11M7D.txt.gz`

for more details see [here](http://www.dei.unipd.it/~capri/BDC/HOMEWORKS/homework4.html)

