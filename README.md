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
      -> validation accuracy: 0.5045294648694082
      -> validation accuracy for top 10 countries:0.6452061945362798
      -> top5 accuracy of validaton: 0.7419294586433397
      -> validation accuracy for tail 10 countries:0.0
  - Conclusion for base model: tail countries are unlearnable due to its small size. The model is underfitting in the top countries, so next step will be increasing the model complexity.
  - Sidetrack: experimented on weights on lightgbm (didn't work as it exagerates the overfitting in rare categories)

3-6 Feb: train a model with two layers (region and country)
  - The idea is to classify the users into regions first then country.
  - grouping the countries into regions according to continent and culture(my instinct)
  - region model worked pretty well in locking in the top 3
  - adjusted hyperparameters based on average users per country in each region (low -> tend to overfit -> smaller tree for generalization)
  - country classifiers given region worked as expected
  - I am currently calculating the probability array user by user which takes a lot of time. I should batch it per region.
  - result:
        -> validation accuracy: 0.5003432351472791
        -> validation accuracy for top 10 countries:0.6451626935792587
        -> validation accuracy for tail 10 countries:0.0
        -> top5 accuracy of validaton: 0.7176110833749376
  - the result is similar to that of base lightgbm.

9-10 Feb: clean up metrics and evaluate the model
  - Because the metrics are slightly lower than flat lightGBM (not what I expected), I spent more time on evaluating the process
  - When I was developing the model, I was just focusing on accuracy and top-k accuracy. They did very good for the region layer (~90% for each region), so I carried on to the next step without careful examination. Now that I added metrics like recall and f1, they look pretty bad for small regions. **I should've done that before carrying on.
  - Also, the fact that the two layer version resulted in very similar accuracy as the flat lightGBM shows that it already captured regional characteristics, making the extra layer redundant.
  - After looking at the second layer (P(country | region, user)), for most countries there is a decent recall and top kaccuracy, meaning that most countries (even with less users) are properly learnt and not ignored.
  - Region by region analysis on the second layer:
        -> Africa(1252): valid acc 0.43, top-k 0.69, recall 0.43, precision 0.25, CE: 2.96 (high), effective classes: 5.81
        -> East Asia(5422) : valid acc 0.50, top-k 0.73, recall 0.496914, CE: 2.69, effective classes: 4.12
        -> West Asia(10079) : valid acc 0.61, top-k 0.83, recall 0.610317, CE: 3.05 (very high), effective classes: 5.93
        -> Nordics(93890) : valid acc 0.58, top-k 0.90,recall 0.579929, CE: 1.92, effective classes: 3.32
        -> Western Core / DACH: valid acc 0.53, top-k 0.86, recall 0.532473, CE: 2.12, effective classes 3.48
        -> Anglo-Europe(21637) : valid acc 0.94, top-k 1.0, recall 0.936473, CE: 0.33, effective classes 1.13
        -> Southern Europe(11384) : valid acc 0.59, top-k 0.91, recall 0.586895, CE: 2.0, effective classes 3.57
        -> Central & Eastern Europe(18517) : valid acc 0.59, top-k 0.79,recall 0.590048, CE: 2.40, effective classes 3.19
        -> Balkans(14738) : valid acc 0.68, top-k 0.86, CE: 1.95, recall 0.676663, effective classes 2.38
        -> Anglo-America(64400) : valid acc 0.88, top-k 0.99, recall 0.879586, CE: 0.57, effective classes 1.27
        -> Latin America(70237) : valid acc 0.83, top-k 0.92, recall 0.825875, CE: 1.53, effective classes 1.77
        -> Oceania(8156): valid acc 0.79, top-k 0.97, recall 0.788610, CE: 1.0, effective classes 1.5
      Conclusion: 1. The model performs better as the CE(cross entropy) decreases. Meaning that regions with more disperse users have similar taste across the region.
                  2. The recall for each region is at least 0.4, showing that small countries are not completey ignored. (Anglo Europe, Central&Eastern Europe, Balkans, Anglo-america, Latin America, Oceania are regions with dominant country but the recall is still good)
                  3. The model is usually uncertain with the ranking but does not ignore tail countries.
    
12 Feb: deploy the model as an endpoint

13-14 Feb: complete the pipeline by fixing the embedding glue job
    
