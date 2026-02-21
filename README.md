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

1-2 Feb: working on training LightGB on sagemaker
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
                The low macro precision also suggests that most countries are often misclassified (assuming as dominant classes).<br/>
                The low macro recall says a similar thing that smaller countries are not learnt by the model.
                The difference between training and validation recall, precision, accuracy also showed that the model tends to overfit in tail countries.<br/>
                simply doing methods like oversampling minorities or SMOTE will not work as it messes with the overfitting and creates an illusion for the model.<br/>
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
    => By only applying weights is not enough to fix the extreme imbalance. Instead, it further exaggerated the problem of overfitting.

3-6 Feb: train a model with two layers (region and country): main goal is to optimize top 5 accuracy and f1 (more balanced)
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

9-10 Feb: clean up metrics and evaluate the model<br/>
  - Because the metrics are slightly lower than flat lightGBM (not what I expected), I spent more time on evaluating the process<br/>
  - metrics for first layer:<br/>
    
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
      1. The aim is to get a high top3 recall so the target country from every region has a chance of standing out in the next stage, not just those from large regions.<br/>
      2. When the number of users in the region get under 10000 the top3 recall gets bad as signals are weak and get surpressed by larger regions. the model results in a high precision low recall. This means that it is very selective on those regions.<br/>
      3. Larger regions have a very high top3 recall
      4. It seems like that regions have high overlap (which make sense as mainstream music has large global influence)<br/>


## Region-Level Country Classification Performance
| Region                   | Entropy | Eff. Classes | Train Acc | Train Prec | Train Rec | Train F1 | Train Top-3 Acc | Train Top-3 Rec | Train Mean True Prob | Val Acc | Val Prec | Val Rec | Val F1 | Val Top-3 Acc | Val Top-3 Rec | Val Mean True Prob |
| ------------------------ | ------- | ------------ | --------- | ---------- | --------- | -------- | --------------- | --------------- | -------------------- | ------- | -------- | ------- | ------ | ------------- | ------------- | ------------------ |
| Africa                   | 2.96    | 5.81         | 0.448     | 0.669      | 0.181     | 0.171    | 0.785           | 0.605           | 0.261                | 0.430   | 0.125    | 0.161   | 0.126  | 0.691         | 0.430         | 0.234              |
| East Asia                | 2.69    | 4.12         | 0.659     | 0.785      | 0.406     | 0.481    | 0.936           | 0.860           | 0.446                | 0.497   | 0.213    | 0.181   | 0.180  | 0.731         | 0.453         | 0.356              |
| West Asia                | 3.05    | 5.93         | 0.758     | 0.891      | 0.516     | 0.622    | 0.955           | 0.865           | 0.531                | 0.610   | 0.226    | 0.211   | 0.213  | 0.826         | 0.329         | 0.451              |
| Nordics                  | 1.92    | 3.32         | 0.799     | 0.891      | 0.649     | 0.716    | 0.990           | 0.955           | 0.548                | 0.580   | 0.349    | 0.336   | 0.314  | 0.897         | 0.653         | 0.451              |
| Western Core / DACH      | 2.12    | 3.48         | 0.724     | 0.891      | 0.531     | 0.612    | 0.978           | 0.932           | 0.485                | 0.532   | 0.275    | 0.228   | 0.220  | 0.861         | 0.481         | 0.389              |
| Anglo-Europe             | 0.33    | 1.13         | 0.939     | 0.469      | 0.500     | 0.484    | 1.000           | 1.000           | 0.904                | 0.936   | 0.468    | 0.500   | 0.484  | 1.000         | 1.000         | 0.890              |
| Southern Europe          | 2.01    | 3.58         | 0.834     | 0.917      | 0.680     | 0.762    | 0.985           | 0.931           | 0.550                | 0.587   | 0.315    | 0.291   | 0.288  | 0.906         | 0.480         | 0.443              |
| Central & Eastern Europe | 2.40    | 3.19         | 0.739     | 0.944      | 0.474     | 0.599    | 0.985           | 0.965           | 0.539                | 0.590   | 0.376    | 0.153   | 0.157  | 0.793         | 0.404         | 0.452              |
| Balkans                  | 1.95    | 2.38         | 0.836     | 0.957      | 0.595     | 0.721    | 0.994           | 0.972           | 0.657                | 0.677   | 0.219    | 0.149   | 0.158  | 0.856         | 0.321         | 0.558              |
| Anglo-America            | 0.57    | 1.27         | 0.884     | 0.983      | 0.436     | 0.549    | 1.000           | 1.000           | 0.813                | 0.880   | 0.126    | 0.143   | 0.134  | 0.997         | 0.495         | 0.796              |
| Latin America            | 1.53    | 1.77         | 0.882     | 0.967      | 0.624     | 0.748    | 0.990           | 0.970           | 0.776                | 0.826   | 0.215    | 0.127   | 0.139  | 0.923         | 0.225         | 0.741              |
| Oceania                  | 1.00    | 1.50         | 0.804     | 0.100      | 0.125     | 0.111    | 0.998           | 0.965           | 0.699                | 0.789   | 0.099    | 0.125   | 0.110  | 0.972         | 0.425         | 0.667              |


<br/>
      Conclusion:<br/> 1. f1 score for each region is better than flat lightgbm -> more balanced <br/>
                      2. The imbalance within region is still huge (neglecting small countries -> dominated by true negatives -> low recall and decent accuracy) <br/>
                      3. Add top k recall: if high-> the model does learn but tail county got dominated, if low -> the model did not learn at all<br/>
                      4. high entropy represents more distributed countries within region, which usually leads to more confusion shown by the low accuracy<br/>
                      5. low entropy represents an imbalance within the region, which usually leads to a high accuracy but low recall<br/>
                      6. Two main problems are overfitting (moderate training recall and bad validation recall) and ignoring minorities
                      7. Anglo-Europe and Oceania are only predicting the dominant country (recall = 1/num_countries)
        

  - When I was developing the model, I was just focusing on accuracy and top-k accuracy. They did very good for the region layer (~90% for each region), so I carried on to the next step without careful examination. Now that I added metrics like recall and f1, they look pretty bad for small regions. **I should've done that before carrying on.
  - Next step: apply weights to both layers (expect to have a lower accuracy in individual models but higher recall)


11 Feb: improving the model <br/>
  - region-level model
 Possible Solutions for problems found yesterday: <br/>
       1. Apply weights (training recall is also low for small regions, the model is not picking up patterns about them, so adding variance might work)<br/>
       2. I came across a 'focal loss' metric for the model to learn (it focuses more on minority examples), but it seems to be very hard to implement <br/>
       3. use SMOTE to try fighting the dominance of large regions (but it creates users linearly which might destroy the pattern as preference of a human is not necessarily linear)<br/>
      4.  apply temperature scaling to flatten out the probabilities, giving small regions a better chance in the following stage. (but it will favor dominant countries of each region)<br/>
    Final Approach:<br/>
        1. apply weights (possibly improve the top3 recall by having decision boundries not ignoring small regions wc​=min(5,sqrt(N / (K*nc)​) -> smaller class size larger weight (with cap and sqrt so it dont go crazy), it might come with a cost of losing some recall for large regions<br/>
        

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

    
    The weighted model definitely helped with recall by sacrificing some recall of the dominant regions. Although this will cause a drop in final accuracy, it results in a more balanced model.<br/>

  - country-level classifier
      -> Solutions: <br/>
            1. use weights to force the model to look at minorities(for Africa, East Asia, Latin-America, Anglo-America, Anglo-Europe and Oceania apply inverse frequence, others apply log frequency which is less aggressive)(less aggressive on regions with moderate train recall->not collapsing  and overfitting)<br/>
            2. to solve overfittng, increase the l2 penalty, min_data_in_leaf and decrease the num_leaves of lightgbm for regions that are overfitting<br/>
      -> Results: <br/>
training accuracy: 0.4703155768349926
validation accuracy: 0.43828409550789155
training accuracy for top 10 countries:0.5931298467783747
validation accuracy for top 10 countries:0.5629893857664868
training accuracy for tail 10 countries:0.15521064301552107
validation accuracy for tail 10 countries:0.0
top5 accuracy of training: 0.7809048806555041
top5 accuracy of validaton: 0.6948603804127883
train Macro recall: 0.14261879644939626
validation Macro recall: 0.06251870706692557
train Macro precision: 0.6366666470772547
validation Macro precision: 0.11361791201995255
train Macro f1: 0.18466604944912227
validation Macro f1: 0.06389570384411046
train accuracy: 0.4703155768349926
validation accuracy: 0.43828409550789155

<b>Wrap up: after trying different methods to try to let the model see more of the rare countries, it failed and bottlenecked at 0.7. Therefore, I believe that the overlap between countries is the root cause and new signals should be introduced to the dataset to boost the hit@5. Since the dataset is provided by kaggle
    
Next Stages:<br/>
 1.deploy endpoint
 2. introduce other embedding methods to include more signals that are not captured by word2vec

    
