# Online Multiclass Boosting
A Python implementation of online multiclass boosting using VFDT as weak learners. 

The algorithms are described and theoretically analazed in the following work. 
```
Young Hun Jung, Jack Goetz, and Ambuj Tewari. 
Online multiclass boosting.
In Advances in Neural Informa- tion Processing Systems, 2017.
```

If you use this code in your paper, please cite the above work. Although it is based on this we cannot guarantee that the algorithm will work exactly, or even produce the same output, as any of these implementations.

For our weak learners, we used the VFDT proposed and implemented by the following two works. 

```
Pedro Domingos and Geoff Hulten. 2000. 
Mining high-speed data streams.
In Proceedings of the sixth ACM SIGKDD international conference on
  Knowledge discovery and data mining (KDD '00). 
ACM, New York, NY, USA, 71-80.
```

```
Vitor da Silva and Ana Trindade Winck. 2017.
Video popularity prediction in data streams based on context-independent features. 
In Proceedings of the Symposium on Applied Computing (SAC '17). 
ACM, New York, NY, USA, 95-100. 
DOI: https://doi.org/10.1145/3019612.3019638
```

The folders "core" and "ht", and the file "hoeffdingtree.py" are copied from the above works, and only minor changes are made to be compatible in python 2.7 (The original code was written in python 3.x). 