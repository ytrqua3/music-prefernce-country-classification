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
  - Conclusion: The recall and tail country accuracy is exteremly low, meaning that the model completely ignored the tail countries.<br/>
                The low precision also suggests that most countries get dominated by the top countries. Small countries are often ignored and misclassified.<br/>
                The difference between training and validation recall, precision, accuracy also showed that the model tends to overfit in tail countries.<br/>
                As a result, simply doing methods like oversampling minorities or other methods like SMOTE will not work.<br/>
                Although underfitting in top countries cost some accuracy, the main problem is the <b>extreme imbalance</b> in classes<br/>
  - Solutions for class imbalance: 1. experimented on weights on lightgbm. But it didn't work as it exagerates the overfitting in rare categories and barely improve the recall<br/>
      *weights are applied on loss function when lightgbm develops a tree, making the model more "focused" on minorities, but because the imbalance is too extreme, it causes more overfitting and has limited benefits to validation metrics<br/>
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
  - adjusted hyperparameters based on average users per country in each region (low -> tend to overfit -> smaller tree for generalization)
  - country classifiers given region worked as expected
  - I am currently calculating the probability array user by user which takes a lot of time. I should batch it per region.
  - Completed the calculation that combine the weighted probabilities by having an initial matrix of probabilities and add probabilities to it from the country model.
  - Overall result:
        -> validation accuracy: 0.5003432351472791<br/>
        -> validation accuracy for top 10 countries:0.6451626935792587<br/>
        -> validation accuracy for tail 10 countries:0.0<br/>
        -> top5 accuracy of validaton: 0.7176110833749376<br/>
  - the result is similar to that of base lightgbm. but need a deeper dive into the metrics to find possible improvements.

9-11 Feb: clean up metrics and evaluate the model
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
      1. In this case, accuracy doesn't mean anything. The aim is to get a high top3 recall so the target country has a chance of standing out in the next stage.<br/>
      2. When the number of users in the region get under 10000 the top3 recall gets bad as signals are weak and get surpressed by larger regions.<br/>
      3. For small regions, the model results in a high precision low recall. This means that it is very selective on those regions.<br/>
      4. It seems like that regions have high overlap (which make sense as mainstream music has large global influence)<br/>
   Possible Solutions: <br/>
       1. Apply weights (training recall is also low for small regions, the model is not picking up patterns about them, so adding variance might work)<br/>
       2. I came across a 'focal loss' metric for the model to learn (it focuses more on minority examples), but it seems to be very hard to implement <br/>
       3. use SMOTE to try fighting the dominance of large regions (but it creates users linearly which might destroy the pattern as preference of a human is not necessarily linear)<br/>
    Final ApproachL<br/>
        1. apply weights (possibly improve the top3 recall by having decision boundries not ignoring small regions wc​=min(5,sqrt(N / (K*nc)​) -> smaller class size larger weight (with cap and sqrt so it dont go crazy)<br/>
        2. apply temperature scaling to flatten out the probabilities, giving small regions a better chance in the following stage.<br/>

      - Comparing performance of weighted and unweighted region model:

## Region-Level Validation Top-3 Recall Comparison

| Region                   | Baseline Val Top3 Rec | Weighted Val Top3 Rec | Δ (Weighted − Base) |
|--------------------------|-----------------------|------------------------|----------------------|
| Africa                   | 0.079                 | 0.1091                 | +0.0301              |
| Anglo-America            | 0.969                 | 0.9153                 | -0.0537              |
| Anglo-Europe             | 0.773                 | 0.7280                 | -0.0450              |
| Antarctica               | 0.000                 | 0.0133                 | +0.0133              |
| Balkans                  | 0.643                 | 0.7012                 | +0.0582              |
| Central & Eastern Europe | 0.725                 | 0.7398                 | +0.0148              |
| East Asia                | 0.363                 | 0.4923                 | +0.1293              |
| Latin America            | 0.959                 | 0.8997                 | -0.0593              |
| Nordics                  | 0.380                 | 0.5195                 | +0.1395              |
| Oceania                  | 0.226                 | 0.4759                 | +0.2499              |
| Southern Europe          | 0.416                 | 0.5413                 | +0.1253              |
| West Asia                | 0.460                 | 0.5437                 | +0.0837              |
| Western Core / DACH      | 0.660                 | 0.6631                 | +0.0031              |

    
    <br/>
    The weighted model definitely helped with recall by sacrificing some recall of the dominant regions.

    

## Region-Level Country Classification Performance

| Region                   | Entropy  | Effective Classes | Train Acc | Train Prec | Train Rec | Train F1 | Train TopK | Val Acc  | Val Prec | Val Rec  | Val F1   | Val TopK |
|--------------------------|----------|------------------|-----------|------------|-----------|----------|------------|----------|----------|----------|----------|----------|
| Africa                   | 2.9609   | 5.8116           | 0.4481    | 0.6690     | 0.1813    | 0.1705   | 0.7851     | 0.4303   | 0.1254   | 0.1610   | 0.1255   | 0.6909   |
| East Asia                | 2.6928   | 4.1160           | 0.6590    | 0.7848     | 0.4062    | 0.4807   | 0.9356     | 0.4969   | 0.2126   | 0.1811   | 0.1802   | 0.7315   |
| West Asia                | 3.0533   | 5.9296           | 0.7580    | 0.8908     | 0.5164    | 0.6220   | 0.9555     | 0.6103   | 0.2263   | 0.2105   | 0.2129   | 0.8262   |
| Nordics                  | 1.9171   | 3.3161           | 0.7991    | 0.8911     | 0.6490    | 0.7165   | 0.9899     | 0.5799   | 0.3490   | 0.3356   | 0.3135   | 0.8970   |
| Western Core / DACH      | 2.1204   | 3.4799           | 0.7240    | 0.8907     | 0.5309    | 0.6125   | 0.9784     | 0.5325   | 0.2750   | 0.2277   | 0.2205   | 0.8611   |
| Anglo-Europe             | 0.3327   | 1.1301           | 0.9387    | 0.4693     | 0.5000    | 0.4842   | 1.0000     | 0.9365   | 0.4682   | 0.5000   | 0.4836   | 1.0000   |
| Southern Europe          | 2.0073   | 3.5770           | 0.8342    | 0.9174     | 0.6805    | 0.7619   | 0.9849     | 0.5869   | 0.3146   | 0.2906   | 0.2884   | 0.9060   |
| Central & Eastern Europe | 2.3977   | 3.1921           | 0.7386    | 0.9443     | 0.4736    | 0.5992   | 0.9855     | 0.5900   | 0.3758   | 0.1532   | 0.1566   | 0.7926   |
| Balkans                  | 1.9547   | 2.3789           | 0.8363    | 0.9565     | 0.5952    | 0.7213   | 0.9937     | 0.6767   | 0.2195   | 0.1491   | 0.1577   | 0.8561   |
| Anglo-America            | 0.5748   | 1.2674           | 0.8838    | 0.9834     | 0.4362    | 0.5491   | 1.0000     | 0.8796   | 0.1257   | 0.1429   | 0.1337   | 0.9967   |
| Latin America            | 1.5258   | 1.7681           | 0.8819    | 0.9669     | 0.6235    | 0.7476   | 0.9902     | 0.8259   | 0.2147   | 0.1271   | 0.1394   | 0.9233   |
| Oceania                  | 1.0005   | 1.4972           | 0.8037    | 0.1005     | 0.1250    | 0.1114   | 0.9979     | 0.7886   | 0.0986   | 0.1250   | 0.1102   | 0.9720   |

<br/>
      Conclusion:<br/> 1. The validation recall is at least 0.12 which is higher than macro recall of flat lightgbm -> this layer does provide a better classification <br/>
                      2. The imbalance within region is still huge (low recall and decent accuracy) <br/>
                      3. Add top k accuracy: if high-> the model does learn but tail county got dominated, if low -> the model did not learn at all
                      4. high entropy represents more evenly distributed countries within region, which usually leads to more confusion shown by the low accuracy
                      5. low entropy represents an imbalance within the region, which usually leads to a high accuracy but low recall

  - When I was developing the model, I was just focusing on accuracy and top-k accuracy. They did very good for the region layer (~90% for each region), so I carried on to the next step without careful examination. Now that I added metrics like recall and f1, they look pretty bad for small regions. **I should've done that before carrying on.
  - Next step: apply weights to both layers (expect to have a lower accuracy in individual models but higher recall)
    
15 Feb: deploy the model as an endpoint (although two methods result in similar result, I deploy the two layer model to challenge myself)

16 Feb: finalize aws glue for embeddings and connect with endpoint


    
