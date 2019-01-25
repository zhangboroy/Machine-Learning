The description of all functions have been included in our report. However, here are some explanations about the main program for our experiment:



1. The dataset
There are 4 datasets used in our experiment, they are all in the “data” folder. In the current code, the TREC data is used. If other datasets are to be used, only the filename needs to be changed (eg. replace "data/TREC" and "data\\TREC" with "data/News-T" and "data\\News-T").



2. The 2 lists of indexes
There are 2 lists of indexes needed (documentStart & domumentEnd) for the experiment. In the current code, the enabled ones are for TREC or TREC-T data, the disabled ones are for News or News-T data. The correct lists must be used, otherwise there will be errors during the running of the program.



3. The outputs
The program will run MStream algorithm first for the dataset, and then run MStreamF algorithm. For either algorithm, it will show the statistic for each batch, including index range of the documents, running time, and NMI. Once all batches are finished, it will show the total NMI of this algorithm for the dataset.