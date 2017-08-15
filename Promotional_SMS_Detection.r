
# AUTHOR: KETAN WALIA
# PROJECT: DETECTING PROMOTIONAL SMS USING TEXT MINING AND MACHINE LEARNING

# CLEARING THE WORKSPACE
rm(list=ls())

# VACATING THE UNUSED MEMORY SPACE
gc()

# READING THE TAB DELIMITED FILE
library(readr)
Spam <- read_delim("~/Desktop/Spam.csv", 
                     "\t", escape_double = FALSE, col_names = FALSE, 
                     trim_ws = TRUE)


# Renaming the Variables.

names(Spam) <- c("Class", "Text_Message")
#View(Spam)

# Summary of the dataset
#summary(Spam)

# Sanity check
sum(which(!complete.cases(Spam)))

# Changing class labels to factor

Spam$Class<- as.factor(Spam$Class)

sapply(Spam, class)

# Plotting the class distribution 

library(ggplot2)

z<-as.data.frame(prop.table(table(Spam$Class)))

ggplot(z, aes(Var1, Freq)) + geom_bar(stat="identity", fill="steelblue") + 
  geom_text(aes(label=round(Freq,2)),vjust=-0.3, size=3.5)+
  theme_minimal()

# Checking distribution of text lengths of the SMS 
# by adding a new feature for the length of each message.
Spam$TextLength <- nchar(Spam$Text_Message)
summary(Spam$TextLength)

Text_length=Spam$TextLength
# Distribution of TextLength
hist(Text_length)

#class(Spam)
#View(Spam)
# Distribution of TextLength by Class
ggplot(Spam, aes(x = TextLength, fill = Class)) + 
  theme_bw() +
  geom_histogram(binwidth = 20) +
  labs(y = "Count", x = "Length of Text",
       title = "Distribution of Text Lengths by Class Labels")


# Splitting data into training and test set using stratified random sampling
library(caret)

set.seed(100)

ind <- createDataPartition(Spam$Class, times = 1,
                               p = 0.7, list = FALSE)

train <- Spam[ind,]
test <- Spam[-ind,]
#dim(train)
#View(train_data)
prop.table(table(Spam$Class))
# Verify proportions.
prop.table(table(train$Class))
prop.table(table(test$Class))

# Building Text Pre-Processing Pipeline

library(quanteda)
library(textstem)

# Lemmatizing the text
train$Text_Message<-lemmatize_strings(train$Text_Message)
#View(spam.raw)

# Tokenizing the text

train.tokens <- tokens(train$Text_Message, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE,remove_separators = TRUE,
                      remove_twitter = TRUE,remove_url = TRUE)

# Lowercasing the Text
train.tokens <- tokens_tolower(train.tokens)

# Removing Stopwords
train.tokens <- tokens_select(train.tokens, stopwords(), 
                             selection = "remove")
# Stemming Tokens
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# Add bigrams to our feature matrix.
#class(train.tokens)
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)

# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.dfm)

# Writing tf , idf functions to perform tf-idf transformation on the dfm matrix

# Function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}


# Normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)

# Calculate the IDF vector that we will use for training and test data!
train.tokens.idf<-apply(train.tokens.matrix, 2, inverse.doc.freq)
length(train.tokens.idf)


# Calculate TF-IDF for our training corpus 
train.tokens.tfidf<- apply(train.tokens.df,2,tf.idf,idf=train.tokens.idf)
#View(head(train.tokens.tfidf))

# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)

# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
length(incomplete.cases)
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))

train.tokens.tfidf.df <- cbind(Label = train$Class, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
# Make a clean data frame.
#train.tokens.tfidf.df <- cbind(Label = train$Class, data.frame(train.tokens.tfidf))
#names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
# Clean up unused objects in memory.
gc()

# Dimensionality Reduction using Latent Semantic Analysis
# Since the size of matrix is huge I will be implementing Truncated SVD to extract the most important features.
library(irlba)

# Time the code execution
start.time <- Sys.time()

# SVD on Traning DATA
# Perform SVD. Specifically, reduce dimensionality down to 100 columns
# for our latent semantic analysis (LSA).
train.irlba <- irlba(t(train.tokens.tfidf), nv = 100, maxit = 600)
#View(train.irlba$v)
dim(train.irlba$v)
train.svd <- data.frame(Label = train$Class, train.irlba$v)

train.svd$TextLength <- train$TextLength

# Setting up parallel computing 
library(doSNOW)
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)
#names(train)
library(caret)

#cv.folds <- createMultiFolds(train$Class, k = 10, times = 3)

#cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                       #  repeats = 3, index = cv.folds)


#stopCluster(cl)
#rf.cv.1 <- train(Label ~ ., data = train.svd, method = "rf", 
 #                                 trControl = cv.cntrl, tuneLength = 7)

# Building Model using Random Forest
library(randomForest)

# Tuning Model Parameter using step Factor=1.5 and threshold value for error improvement=0.01

mtry <- tuneRF(train.svd[,2:102],train.svd$Label, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
#best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)

# As a result of parameter tuning in the previous step the value of mtry=15
# gives the lowest OOBV error. Hence, we use mtry=15 to build model on traning data
rf<-randomForest(train.svd[,c(2:102)],train.svd$Label,mtry=15)
rf
confusionMatrix(train.svd$Label, rf$predicted)
plot(rf$err.rate[, 1], type = "l", xlab = "number of trees", ylab = "OOB error")

importance(rf)
varImpPlot(rf)

# Testing the model on the test data 
# Pre processing test data using the similar text processing pipeling as training data
#      0 - Lemmatization
#      1 - Tokenization
#      2 - Lower casing
#      3 - Stopword removal
#      4 - Stemming
#      5 - Adding bigrams
#      6 - Transform to dfm
#      7 - Ensure test dfm has same features as train dfm


# Lemmatization
train$Text_Message<-lemmatize_strings(train$Text_Message)
# Tokenization.
test.tokens <- tokens(test$Text_Message, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)
# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

# Stemming.
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# Add bigrams.
test.tokens <- tokens_ngrams(test.tokens, n = 1:2)

# Convert n-grams to document-term frequency matrix.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)

# Ensure the test dfm has the same n-grams as the training dfm.

test.tokens.dfm <- dfm_select(test.tokens.dfm, features = train.tokens.dfm)
test.tokens.matrix <- as.matrix(test.tokens.dfm)

# Projecting the term counts for the n grams into the same TF-IDF vector space as our training
# data. The process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)

# Lastly, calculate TF-IDF for our training corpus.
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)

# Transpose the matrix
test.tokens.tfidf <- t(test.tokens.tfidf)

# Fix incomplete cases
#summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
#summary(test.tokens.tfidf[1,])


# With the test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)

test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))

# Adding Label and TextLength to test data
test.svd <- data.frame(Label = test$Class, test.svd.raw, 
                       TextLength = test$TextLength)
# Making Prediction on test data
preds <- predict(rf, test.svd)

# Getting confusion matrix
confusionMatrix(preds, test.svd$Label)
