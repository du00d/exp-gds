# exp-gds

The following is a quick-test to measure the total time to perform "NUM_READS" over fixed size array of length "ARRAY_SIZE", using gds and mmap.

It has been tested with the features from Products graph in the Stanford OGB Repository.
```
get_dataset.py
```
will download the zip file, extract the features. It will take < 5 minutes.


To compile all the experiments. This will output gds_read and mmap_read.

```
make 
```

To execute the program. (Assume we store feat.bin file on /mnt/nvme/)

parameter 1: /mnt/nvme/feat.bin

parameter 2: <NUM_READS>

parameter 3: <ARRAY_SIZE>


For example: 
```./gds_read /mnt/nvme/feat.bin 10000 100```

Similarly,
```./mmap_read /mnt/nvme/feat.bin 10000 100```


