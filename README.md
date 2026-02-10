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
      -> training accuracy: 0.6598848248296058
      -> validation accuracy: 0.5045294648694082
      -> training accuracy for top 10 countries:0.6984821241437128
      -> validation accuracy for top 10 countries:0.6452061945362798
      -> top5 accuracy of training: 0.9347353750631271
      -> top5 accuracy of validaton: 0.7419294586433397
      -> training accuracy for tail 10 countries:0.9002217294900222
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
        -> training accuracy: 0.5515714952849256
        -> validation accuracy: 0.5003432351472791
        -> training accuracy for top 10 countries:0.6830441353292224
        -> validation accuracy for top 10 countries:0.6451626935792587
        -> training accuracy for tail 10 countries:0.07982261640798226
        -> validation accuracy for tail 10 countries:0.0
        -> top5 accuracy of training: 0.8218082497125433
        -> top5 accuracy of validaton: 0.7176110833749376
  - the result is similar to that of base lightgbm.

9-10 Feb: finalize and evaluate the model
  - Transfer from jupyterlab to sagemaker notebook
  - use framework estimator (sagenaker.SKLearn) to launch training job

12 Feb: deploy the model as an endpoint

13-14 Feb: complete the pipeline by fixing the embedding glue job
    
