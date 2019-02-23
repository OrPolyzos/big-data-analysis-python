# Big Data Analysis in Python
The full datasets can be found under the `resources/datasets/full/`

The minified datasets can be found under the `resources/datasets/min/`

Basic configuration options are exposed through `resources/configs.cfg`

## Data structure
Working with articles.
```
[Id],[Title],[Content],[Category]
```

## Functionalities

### Word Cloud
Based on the `Content` column of the top category (most entries) using **WordCloud** (from wordcloud library)

*Sample*
![sample-word-cloud](resources/datasets/min/word-cloud.png?raw=true "Word Cloud")

### Deduplication
Based on the `Content` column using **TfidfVectorizer** (from sklearn library)

*Sample tsv*
```
	ID_1	ID_2	Similarity
0	11214	11195	0.7042086714029414
1	11293	11195	0.7297366699959508
2	11195	11214	0.7042086714029414
3	11195	11293	0.7297366699959508
```

### Classification
Based on the `Content` column and using 

* Support Vector Machines (SVM) - **SGDClassifier** (from sklearn library)
  * CountVectorizer
  * TruncatedSVD

* Random Forests - **RandomForestClassifier** (from sklearn library)
  * CountVectorizer
  * TruncatedSVD
  
Metrics are being exported from the above stages.
*Sample tsv*
```
classifier	feature	accuracy	precision	recall	f-measure	auc
SVM	BoW	0.9236334918582818	0.93037234987235	0.9148975468975469	0.9225700609862193	0
SVM	SVD	0.4523594053005818	0.3694365612454068	0.4352236652236653	0.39964081475333885	0
Random Forest	BoW	0.8037999815310739	0.8243787780846603	0.7676580086580087	0.7950079752355407	0
Random Forest	SVD	0.6713206436420722	0.6514912675044254	0.6474978354978356	0.6494884130742743	0
Custom Random Forest	SVD	0.7054410225628713	0.7018007635639213	0.6887532467532468	0.6952157929749285	0
```

Predictions are finally done with a parameterized **RandomForest** using **TruncatedSVD**
*Sample tsv*
```
Test_Document_ID	Predicted_Category
2	Politics
10	Technology
25	Film
28	Technology
```