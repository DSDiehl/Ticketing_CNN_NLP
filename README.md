# Sales Opportunity Data Analysis
##  Marcus Diehl, MSDS 696, Regis University 
### Dec 12, 2018


# Introduction
My project is titled 'NLP Auto Routing' which is the application of Natural Language Processing (NLP) techniques combined with Tensorflow as an application to route support tickets to the appropriate resolving teams simply by the contents of the description with in the ticket.  The ideal for this project was based on an article I had discovered by Uber Engineers (https://eng.uber.com/nlp-deep-learning-uber-maps/) where they wrote how they were implementing something similar with in the Uber in-house customer support platform.  

The article would go on to describe how people would comment that the driver was excellent or they left an item in the car and needed help retrieving this.  Before the article, all submitted issues were read and routed to the appropriate team manually.  The Uber engineers decided to develop a Neural Network model based on NLP to auto route the issues to appropriate teams.    Knowing my employer had a similar 'in-hour support platform', I figured I could give it a try to recreate the article based simply on the terminology as the Uber article provided zero code.    

Basically my thought process was if they could do it, so could I. Boy was I naive.  Before this project, I had no experience with Neural Networks, no experience with Tensorflow and very little experience with Python.   Each of these topics alone requires a huge amount of comprehension.  On top this, the Uber article mentions the use of 2 Neural Networks. 1 is for the Word-To-Vector (Word2Vec) process of sentence comprehension while the other was a Convolutional Neural Network (CNN) which is typically used in Image recognition. 


# The Data
The data available was hosted on a SQL server and was complex its relationships with other tables.  With some help from an admin, I was able to identify the proper table that contained both the text string from the user opening the Ticket, along with the team that was able to properly resolve the ticket.   Normally, the Service Desk would receive the ticket and either resolve internally or forward the ticket to another team to resolve.  It was later discovered that the Service Desk actually self-resolved nearly 85% of all the tickets. Also these are technical tickets, with issues related to Firewall Configurations, Access Permissions, Laptop issues, or even security events requiring IT Security involvement. 

Below are  few examples taken of the data to understand the context of the topics included in the technical tickets.  This is the most I am allowed the shared but can say each one was resolved by a different team. 

![Image](/Images/image5.png)

The data set I was able to extract contained close to 600,000 samples (nearly 6 years of history) and involved 478 unique resolving teams.  Though the data contained 12 features (Ticket ID, Date, ID person opened ticket, etc…), the samples were simplified to include 2 features for this project; The String of Text for Ticket Description and the ID of team whole resolved the ticket.  

Even then more preprocessing was required of the data.  Through Exploratory Data Analysis, it was discovered that some of the ticket descriptions were over 33k words long.  Previewing these tickets found that some individual would copy an email chain or include the contents of a log file directly into the ticket description.  Also for the 478 resolving teams, there were many that only had a single ticket resolution.   In the end, it was decided to take action to simplify the data.  For the resolving teams,  only tickets from the top 10 were kept.  

Second, the Ticket Description would have punctuation, numbers and stop words removed. Then reduced by only keeping the first 100 words.   The first 100 words was arbitrary number but it was still felt that the purpose of the ticket would be stated with in those first 100 words.  

# 1st NN (Word2Vec)
Once the data was prepared, a Word2Vec Neural Network was generated as a way to measure the similarity between words and their usage.  A Word2Vec is a more advanced method with in NLP research over a Bag of Words or TF-IDF. The main difference is that a Word2Vec produce a vector which can be used to identify the content of document along with subcontexts.  Bag of Words simply provides the frequency of the ngram while TF-IDF can help identify individual important ngrams but neither can show the relationship between words in terms or interchangeability.  This is important later on with the Convolution Neural Network (CNN) in trying to find the relationships between words and their importance towards a resolving team.  The Word2Vev also helps in realizing that both 'password'  and 'passwd' are similar but then so is 'John' and 'Micheal' or 'Nov' and 'Dec'. 

Below is a scatter plot of the top 500 words from the custom Word2Vec model that was built for this project. Any more words actually created too much overlap in the plot to allow for much comprehension. Word2Vec uses multiple dimensions to represent a word which was reduced to 2 dimensions using a Principal Component Analysis (PCA).  The results are very interesting simply due to how surprisingly effective a Word2Vec is which is mainly due to how structured a sentence is.

![Image](/Images/image1.png)

There are plenty of existing Word2Vec Tensorflow models already available to be reused and some even exist within Tensorflow itself under the now depreciated estimator module. However, most of these models are simply built off of Wikipedia since it is the large publicly accessible data source for NLP.  Being that the data used in this project comes from a more technical based solution, a custom Word2Vec was generated from that source data to create a more accurate mapping of relationships and thus greatly improve the CNN accuracy. 

With this project, the Word2Vec used a vector size of 100 dimensions to represent each word, and the dictionary size was limited to 50,000 words based on occurrence frequency.  The dictionary limit was done after the data preprocessing, so stopwords were already removed and not included in the Word2Vec dictionary.   The ticket descriptions were then all combined  into a single string.  Any words the existed outside of those 50,000  word dictionary size limit were replaced with the ID related to 'UNK'.   

The actual Word2Vec NN is very basic, well as basic a Neural Network can be, with no hidden layers.    Below is a Tensorboard output from a Summary Writer of the Word2Vec TF NN.

![Image](/Images/image2.png)

# 2nd NN (CNN)
After the Word2Vec NN, a Convolution Neural Network (CNN) is then used to build the actual prediction model against the Ticket Descriptions and producing a Resolving Team ID as output.   An initial step before building out the CNN was to actually pad the ticket descriptions that were already less than 100 words.  NN require consistency, so any ticket with less than 100 words had the UNK word added till the ticket reached 100 words.

In the end, the CNN was the hardest to understand and actually took the longest portion of the project to get properly implemented.  Below is a helpful visual example I found that explains how different ngram layers a built and then combined by multiple hidden layers using pooling and softmax.   However, this projects implementation was on a whole different scale in comparing to the example in the visual.

![Image](/Images/image3.png)

Image Reference : http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

Overall, the CNN size was enormous. With 100,000 input neurons (due to 100 words per ticket x 100 dimensions per Word2Vec embedding) along with the multiple hidden layers (for different ngram layers, pooling and softmax normalizations), the project was migrated off a personal laptop due to lack of resources.   In the end, an Azure Cloud server was purchased with a NVidia card included. This dramatically sped up batch processing (from 50 samples a sec to 5000 samples a sec). 

Below is the Tensorboard design output for the CNN, including the 4 different ngram convolutional 2d mappings.

![Image](/Images/image4.png)

Tensorflow's workhorse for convolution is the conv2d function which typically is applied to images, has an 2 array input.  Again, images having an x and y pixel coordinate system, conv2d uses those coordinate as input.   But with words, it is different.  Here we used Y as the embedding per word, and the X as the words in the ticket description.   With an embedding size coming from the Word2Ve, our Y length is 100.    But with X, using the Convolution feature to pool together different ngrams as features with in the CNN, we combine the different pools to create different points of view (features) against the sentences . 

Again, I am following the Uber example , but think like Hypermeters or Data Set sizes used were not shared.  These I had to learn and experiment with myself.  Things like deciding on a type of data padding method to use (Narrow or Wide), or the Stride size (how much to move the collusion window).  But one thing the article did theorize was the natural migration from a Word based CNN to a Character level CNN.  The thought process was that a Character level would allow for the CNN to be aware of word misspellings.

# References
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

https://www.tensorflow.org/guide/

https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f

http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/

http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

