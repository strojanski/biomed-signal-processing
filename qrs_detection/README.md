## QRS detection
This folder contains implementation of the ECG QRS detection, described in the `Ding_ecg.pdf` paper. 

### Instructions
To fully complete the assignment you have to satisfy the following requirements: 
1. The QRS detector has to accept the name of the record (.dat and .hea files of the database) as a parameter. 
2. The program has to open this record and detect the QRS complexes in the record. 
3. The program has to write an annotation file (.asc) in the ASCII format, which contains sample indexes of the fiducial points (FPs) of the detected QRS complexes. 
4. The performance of the program has to be evaluated in the sense of sensitivity (Se) and positive predictivity (+P) with regard to manually annotated fiducial points (FPs) of the QRS complexes (.atr files of the database). 
5. The results of the performance evaluation together with your discussions and conclusions have to be written into another document (.pdf) file. Discuss also the weaknesses of the implemented technique and possible improvements. 
6. The .asc and .pdf files, together with the source of your program (included in a single compressed .zip archive) have to be submitted while uploading the assignment.

### Data
[Long-Term ST Database](https://www.physionet.org/content/ltstdb/1.0.0/)  \
[MIT/BIH Arrhytmia Database](https://www.physionet.org/content/mitdb/1.0.0/) 

### Additional literature
[General](https://lbcsi.fri.uni-lj.si/OBSS/vaje/literaturaSplosnaECG.html) \
[WFDB](https://lbcsi.fri.uni-lj.si/OBSS/vaje/literaturaWFDB.html) 
