# music-prefernce-country-classification
This project is an extension to the term-project-3280.

link to project: https://github.com/ytrqua3/news_aggregator_project

After taking CPSC3280 in Langara, I found that the course mainly focuses on introducing different cloud services (e.g. ec2, s3, load balancer, Kinesis, Aurora, DynamoDB...) and building data ETL pipelines using AWS Glue and pyspark. In the term project, I deployed a pipeline that aggregates data and api gateways to fetch the data. Within the project, spark was used to form embeddings for each artist using a simple neural network model called word2vec. Later, an idea that I could predict users' country according to their preference in artists popped up. 


Log of notes and learning:

14-17 Jan: working on the music embedding notebook that transforms every user into an embedding
  1. import data
  2. clean up the data: drop null, cast types, fill up missing playcounts
  3. for every user, create a sequence that repeats the artists' names log(playcount) times
  4. fit the df into a w2v model to get an embedding per artist
  5. store artist embeddings to s3
  6. form user embeddings (weighted mean of all artist embeddings): add a column for artist embeddings to top_artist_df -> scale each vector with log(playcount) -> sum vectors up per user -> take the weighted mean of the vector
  7. clean the data more
  8. filter out rows with null country, null total_scrobbles, and countries with less than 50 users
  9. do train valid test split

19-20 Jan: worked on applying logistic regression
  - used vector assembler -> label indexer -> logistic regression as pipeline and resulted in a 47% accuracy
  - used AWS Glue as spark has its own logistic regression class and the process can be parallelized
  - tried to apply LightGB but could not figure out how to train one
  - final: completed the logistic regression as a baseline and went on a side track to taking a crash course on sagemaker

1-2 Feb: working on training and deploying LightGB on sagemaker
  - base model did not work very well
training accuracy: 0.6598848248296058 <br/>
validation accuracy: 0.5045294648694082<br/>
training accuracy for top 10 countries:0.6984821241437128<br/>
validation accuracy for top 10 countries:0.6452061945362798<br/>
training accuracy for tail 10 countries:0.9002217294900222<br/>
validation accuracy for tail 10 countries: <mark>0.0</mark><br/>
top5 accuracy of training: 0.9347353750631271<br/>
top5 accuracy of validaton: 0.7419294586433397<br/>
train Macro recall: 0.7685098271292383<br/>
validation Macro recall: <mark> 0.07325228841703592</mark><br/>
train Macro precision: 0.957286232098812<br/>
validation Macro precision: <mark>0.15461463815748647</mark><br/>
train Macro f1: 0.8311340806063884<br/>
validation Macro f1: 0.0853215936831941<br/>
train accuracy: 0.6598848248296058<br/>
validation accuracy: 0.5045294648694082<br/>
  - Conclusion: The recall and tail country accuracy is exteremly low, meaning that the model completely ignored the tail countries.
                The low precision also suggests that most countries get dominated by the top countries. Small countries are often ignored and misclassified.
                The difference between training and validation accuracay also showed that the model tends to underfit in top countries and overfit in tail countries.
                Although overfitting and underfitting cost some accuracy, the main problem is the <b>extreme imbalance</b> in classes
  - Solution: 1. experimented on weights on lightgbm. But it didn't work as it exagerates the overfitting in rare categories and barely improve the recall
              2. Soft hierarchy: P(country | user) = P(country | user, region) * P(region | user). Expects to lower the recall by giving tail countries a chance to be captured by the model (increase its effective sample size)
  - results for applying weights to lightgbm
training accuracy: 0.6389129217771897<br/>
validation accuracy: 0.4809015347258973<br/>
training accuracy for top 10 countries:0.6389143216440714<br/>
validation accuracy for top 10 countries:0.595876109274404<br/>
training accuracy for tail 10 countries:0.991130820399113<br/>
validation accuracy for tail 10 countries:<mark>0.0</mark><br/>
top5 accuracy of training: 0.8896988298484558<br/>
top5 accuracy of validaton: 0.7281698471500171<br/>
train Macro recall: 0.8353119347987866<br/>
validation Macro recall: <mark>0.0876825448523376</mark><br/>
train Macro precision: 0.8492974903657015<br/>
validation Macro precision: <mark>0.1004111028582959</mark><br/>
train Macro f1: 0.8332193850073149<br/>
validation Macro f1: 0.08817902302231734<br/>
train accuracy: 0.6389129217771897<br/>
validation accuracy: 0.4809015347258973<br/>
    => By only applying weights is not enough to fix the extreme imbalance. Instead, it further exaggerated the problem of overfitting of tail countries.

3-6 Feb: train a model with two layers (region and country)
  - The idea is to classify the users into regions first then country.
  - grouping the countries into regions according to continent and culture(my instinct)
  - region model worked pretty well in locking in the top 3
  - adjusted hyperparameters based on average users per country in each region (low -> tend to overfit -> smaller tree for generalization)
  - country classifiers given region worked as expected
  - I am currently calculating the probability array user by user which takes a lot of time. I should batch it per region.
  - Completed the calculation that combine the weighted probabilities.
  - Overall result:
        -> validation accuracy: 0.5003432351472791
        -> validation accuracy for top 10 countries:0.6451626935792587
        -> validation accuracy for tail 10 countries:0.0
        -> top5 accuracy of validaton: 0.7176110833749376
        -> validation recall: 0.1
        -> validation precision: 0.1
  - the result is similar to that of base lightgbm.

9-10 Feb: clean up metrics and evaluate the model
  - Because the metrics are slightly lower than flat lightGBM (not what I expected), I spent more time on evaluating the process
  - metrics for first layer:
    | Region                   | Users (True) | Users (Pred) | Train Acc | Train Prec | Train Rec | Train F1 | Train Top3 Rec | Val Acc | Val Prec | Val Rec | Val F1 | Val Top3 Rec |
    | ------------------------ | ------------ | ------------ | --------- | ---------- | --------- | -------- | -------------- | ------- | -------- | ------- | ------ | ------------ |
    | Africa                   | 1252         | 332          | 0.996     | 0.967      | 0.256     | 0.405    | 0.595          | 0.995   | 0.600    | 0.036   | 0.069  | 0.079        |
    | Anglo-America            | 64400        | 103843       | 0.778     | 0.537      | 0.866     | 0.663    | 0.977          | 0.756   | 0.513    | 0.834   | 0.635  | 0.969        |
    | Anglo-Europe             | 21637        | 9926         | 0.921     | 0.575      | 0.264     | 0.362    | 0.828          | 0.912   | 0.461    | 0.205   | 0.284  | 0.773        |
    | Antarctica               | 616          | 257          | 0.999     | 1.000      | 0.417     | 0.589    | 0.859          | 0.998   | 0.000    | 0.000   | 0.000  | 0.000        |
    | Balkans                  | 14738        | 12881        | 0.951     | 0.585      | 0.511     | 0.545    | 0.754          | 0.939   | 0.462    | 0.390   | 0.423  | 0.643        |
    | Central & Eastern Europe | 18517        | 16081        | 0.934     | 0.552      | 0.480     | 0.513    | 0.788          | 0.922   | 0.444    | 0.397   | 0.419  | 0.725        |
    | East Asia                | 5422         | 3046         | 0.984     | 0.700      | 0.393     | 0.504    | 0.605          | 0.980   | 0.507    | 0.231   | 0.318  | 0.363        |
    | Latin America            | 70237        | 82388        | 0.879     | 0.739      | 0.867     | 0.798    | 0.967          | 0.866   | 0.713    | 0.853   | 0.777  | 0.959        |
    | Nordics                  | 9389         | 4081         | 0.971     | 0.755      | 0.328     | 0.458    | 0.551          | 0.968   | 0.612    | 0.227   | 0.332  | 0.380        |
    | Oceania                  | 8156         | 364          | 0.969     | 0.901      | 0.040     | 0.077    | 0.482          | 0.968   | 0.545    | 0.006   | 0.011  | 0.226        |
    | Southern Europe          | 11384        | 4004         | 0.962     | 0.717      | 0.252     | 0.373    | 0.547          | 0.958   | 0.554    | 0.172   | 0.263  | 0.416        |
    | West Asia                | 10079        | 4677         | 0.970     | 0.759      | 0.352     | 0.481    | 0.607          | 0.965   | 0.633    | 0.264   | 0.373  | 0.460        |
    | Western Core / DACH      | 19610        | 13557        | 0.928     | 0.549      | 0.380     | 0.449    | 0.768          | 0.913   | 0.437    | 0.300   | 0.356  | 0.660        |
    <br/>
    Conclusion:<br/>
      1. Tail reigions have very low recalls and still very high accuracy -> they are ignored and because of their small size, accuracy got dominated by the true negatives.<br/>
      2. For most small regions, they have a low recall high precision -> a group of users have unique music taste (easily identifiable) and the model can only that particular group of users in the small countries<br/>

    | Region                   |       Entropy | Effective Classes | Train Acc | Train Prec | Train Rec | Train F1 | Train TopK | Val Acc  | Val Prec | Val Rec  | Val F1   | Val TopK |
    | ------------------------ | ------------- | ----------------- | --------- | ---------- | --------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- |
    | Africa                   | 2.960907      | 5.811554          | 0.448083  | 0.668985   | 0.181254  | 0.170543 | 0.785144   | 0.430303 | 0.125391 | 0.160963 | 0.125511 | 0.690909 |
    | East Asia                | 2.692764      | 4.116009          | 0.658982  | 0.784769   | 0.406175  | 0.480654 | 0.935633   | 0.496914 | 0.212591 | 0.181096 | 0.180247 | 0.731481 |
    | West Asia                | 3.053301      | 5.929560          | 0.758012  | 0.890823   | 0.516358  | 0.621950 | 0.955452   | 0.610317 | 0.226298 | 0.210517 | 0.212880 | 0.826190 |
    | Nordics                  | 1.917101      | 3.316114          | 0.799127  | 0.891092   | 0.648951  | 0.716483 | 0.989882   | 0.579929 | 0.348970 | 0.335584 | 0.313519 | 0.896980 |
    | Western Core / DACH      | 2.120414      | 3.479939          | 0.724018  | 0.890680   | 0.530935  | 0.612482 | 0.978378   | 0.532473 | 0.274998 | 0.227689 | 0.220476 | 0.861111 |
    | Anglo-Europe             | 0.332703      | 1.130119          | 0.938670  | 0.469335   | 0.500000  | 0.484182 | 1.000000   | 0.936473 | 0.468237 | 0.500000 | 0.483597 | 1.000000 |
    | Southern Europe          | 2.007283      | 3.576967          | 0.834153  | 0.917352   | 0.680497  | 0.761887 | 0.984891   | 0.586895 | 0.314577 | 0.290619 | 0.288420 | 0.905983 |
    | Central & Eastern Europe | 2.397691      | 3.192067          | 0.738565  | 0.944253   | 0.473610  | 0.599212 | 0.985473   | 0.590048 | 0.375788 | 0.153169 | 0.156613 | 0.792602 |
    | Balkans                  | 1.954688      | 2.378896          | 0.836341  | 0.956539   | 0.595196  | 0.721304 | 0.993690   | 0.676663 | 0.219463 | 0.149057 | 0.157666 | 0.856052 |
    | Anglo-America            | 0.574772      | 1.267369          | 0.883820  | 0.983358   | 0.436193  | 0.549133 | 1.000000   | 0.879586 | 0.125655 | 0.142857 | 0.133705 | 0.996709 |
    | Latin America            | 1.525788      | 1.768083          | 0.881900  | 0.966901   | 0.623508  | 0.747568 | 0.990205   | 0.825875 | 0.214704 | 0.127066 | 0.139374 | 0.923330 |
    | Oceania                  | 1.000518      | 1.497208          | 0.803703  | 0.100463   | 0.125000  | 0.111396 | 0.997916   | 0.788610 | 0.098576 | 0.125000 | 0.110227 | 0.972008 |
<br/>
      Conclusion:<br/> 1. The validation recall is at least 0.12 which is higher than macro recall of flat lightgbm -> this layer does provide a better classification <br/>
                      2. The imbalance within region is still huge (low recall and decent accuracy) <br/>
                      3. Add top k accuracy: if high-> the model does learn but tail county got dominated, if low -> the model did not learn at all

  - When I was developing the model, I was just focusing on accuracy and top-k accuracy. They did very good for the region layer (~90% for each region), so I carried on to the next step without careful examination. Now that I added metrics like recall and f1, they look pretty bad for small regions. **I should've done that before carrying on.
  - Next step: apply weights to both layers (expect to have a lower accuracy in individual models but higher recall)
    
15 Feb: deploy the model as an endpoint (although two methods result in similar result, I deploy the two layer model to challenge myself)

16 Feb: finalize aws glue for embeddings and connect with endpoint


    
