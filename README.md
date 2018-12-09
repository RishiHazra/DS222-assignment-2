# DS222-assignment-2
Assignment 2 (Distributed Logistic regression using tensorflow)


## To run this.......

### for local code

```
cd ds222/assignment2   
python example.py
```


### for distributed code (change the nodes accordingly)

```
cd ds222/assignment2  
pc-01$ python example.py --job_name="ps" --task_index=0     
pc-02$ python example.py --job_name="worker" --task_index=0     
pc-03$ python example.py --job_name="worker" --task_index=1     
pc-04$ python example.py --job_name="worker" --task_index=2    
```
Similarly for Drop Stale Synchronous (poincare_stale.py)

### References
[Distibuted tensorflow example] (https://github.com/ischlag/distributed-tensorflow-example)  
[Distributed tensorflow documentation] (http://www.tensorflow.org/deploy/distributed)
