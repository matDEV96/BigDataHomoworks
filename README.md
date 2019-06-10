# BigDataHomoworks
## Homeworks for the course on Big Data Computing, held @University of Padua

## Homework 4 

### Accesing the cluster:

putty: `gajdusek@login.dei.unipd.it`

when the connection opens: `ssh -p 2222 group50@147.162.226.106` (insert the new password...)

****TODO****: uploading and running jobs:

According to the webpage of hw:

- upload the code using `scp` (linux) or `pscp` (windows)
- `spark-submit --conf spark.pyspark.python=python3 --num-executors X GxxHM4.py argument-list `
- maximum X: 32
- to pass one of preloaded files as an argument: specify path `/data/filename`, e.g. `/data/HIGGS11M7D.txt.gz`

for more details see [here](http://www.dei.unipd.it/~capri/BDC/HOMEWORKS/homework4.html)

