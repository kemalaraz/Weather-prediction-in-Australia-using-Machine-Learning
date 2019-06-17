#####REQUIRED LIBRARIES#####
install.packages("VIM")
install.packages("caret")
install.packages("RANN")

library(caret)
library(VIM)
library(RANN)

#####READ TRAIN AND TEST DATASET#####

df_train <- read.csv("rain_to_R_train.csv", sep=",", strip.white = TRUE)
df_test <- read.csv("rain_to_R_test.csv", sep=",", strip.white = TRUE)
df_train <- df_train[,-1] #To drop the index column
df_test <- df_test[,-1] #to drop the index column

#####PERCENTAGE OF NA#####

# To see which attribute has how many missing values in percentage, nans dataframe created
nans=as.data.frame(matrix(ncol = 1, nrow = 19), stringsAsFactors = F)
for (i in 1:length(df_train)) {
  nans[i,1] <- sum(is.na(df_train[,i]))/length(df_train[,i])
}

colnames(nans) <- "Percentage of NA" # Assigning a column name
rownames(nans) <- colnames(df_train) # Assigning row names

nans <- nans[order(nans),,drop=F] #Sorted
nans

#####LISTWISE DELETION OF SOME ATTRIBUTES NA VALUES#####

# Since a lot of the attributes has less than %5 of NA values listwise deletion can be used to delete them(Shafer, 1999).
na <- nans[which(nans<=0.05 & nans>0),,drop=F] # Attributes that have less than %5 and more than 0 missing values

df_train_complete <- df_train[complete.cases(df_train[,c("Temp3pm","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm")]),]
df_train_complete # Missing values of the attributes which has less than %5 missing values, listwise deleted

# To check if those attributes still contain missing values or not - nans_complete dataframe created
nans_complete=as.data.frame(matrix(ncol = 1, nrow = 19), stringsAsFactors = F)
for (i in 1:length(df_train_complete)) {
  nans_complete[i,1] <- sum(is.na(df_train_complete[,i]))/length(df_train_complete[,i])
}

colnames(nans_complete) <- "Percentage of NA" # Assigning a column name
rownames(nans_complete) <- colnames(df_train) # Assigning row names
nans_complete <- nans_complete[order(nans_complete),,drop=F] #Sorted
nans_complete

#####MCAR(mising completely at random) CHECK##### 

#For attributes which still has missing values, graphs created and analyzed below
# Continuous attribtues compared with continuous attributes
# WindGustSpeed
marginplot(df_train_complete[c("WindGustSpeed","Pressure9am")]) 
marginplot(df_train_complete[c("WindGustSpeed","Evaporation")])
marginplot(df_train_complete[c("WindGustSpeed","Sunshine")]) # It seems WindGustSpeed might be MCAR
pbox(df_train_complete,pos=7) # It seems our first opposition is true and WindGustSpeed is MCAR

marginplot(df_train_complete[c("Pressure9am","Evaporation")])
marginplot(df_train_complete[c("Pressure9am","Sunshine")]) # It seems Pressure9am might be MCAR
pbox(df_train_complete,pos=14) # It seems our first opposition is true and Pressure9am is MCAR

marginplot(df_train_complete[c("Evaporation","Sunshine")]) # It seems that Evaporation might be MCAR
pbox(df_train_complete,pos=4) # It seems our first opposition is true and Evaporation is MCAR
pbox(df_train_complete,pos=5) # Sunshine also seems MCAR


# Categorical attribtues compared with categorical attributes
marginplot(df_train_complete[c("Cloud3pm","Cloud9am")]) # It seems Cloud9am MCAR but Cloud3pm may be MAR
pbox(df_train_complete,pos=15) # It seems our first opposition is true and Cloud9am is MCAR
pbox(df_train_complete,pos=16) # It seems our first opposition may be wrong because those instances with missing information for other attributes are not much higher or lower than those of the non-missing instances.
# Only differences occured are the comparision with attributes also has missing values and even those differences are not much.


# One study showed that listwise
# deletion leads to a decrease in statistical power 
# if more than 10% of the data is missing(Raaijmakers, 1999). 
# Since WindGustSpeed and Pressure9am is MCAR and missing values are less then %10, listwise deletion is possible.
df_train_complete <- df_train_complete[complete.cases(df_train_complete[,c("Pressure9am","WindGustSpeed")]),]

#####IMPUTATION#####

# Since all the attributes' missing values are MCAR there is no need to dig more and imputation models that can deal with MCAR can easily be implemented
# For this dataset bagging ensemble algoritm imputation method chosen because it can deal with both categorical and numerical values also it doesn't need scaling which is going to be the next part in the process.
# Since this data is not completely time series data but has some temporal aspects in it and also imputation of Cloud9am, Cloud3pm, Evaporation and Sunshine doesn't need Location and Date  attribute to be imputed
# they are removed before the imputation process.
date_loc <- df_train_complete[,c(1,2)]
df_train_complete <- df_train_complete[,-c(1,2)]

# Train the imputer
df_train_imp <- preProcess(df_train_complete, method = "bagImpute", lev=NULL)
# Apply the imputer to the training dataset
df_train_imputed <- predict(df_train_imp, newdata = df_train_complete)

# To check training dataset if those attributes still contain missing values or not nans_train_imp dataframe created
nans_train_imp=as.data.frame(matrix(ncol = 1, nrow = 17), stringsAsFactors = F)
for (i in 1:length(df_train_imputed)) {
  nans_train_imp[i,1] <- sum(is.na(df_train_imputed[,i]))/length(df_train_imputed[,i])
}

colnames(nans_train_imp) <- "Percentage of NA" # Assigning a column name
rownames(nans_train_imp) <- colnames(df_train_complete) # Assigning row names
nans_train_imp <- nans_train_imp[order(nans_train_imp),,drop=F] #Sorted
nans_train_imp

# Apply listwise deletion (as decided on training dataset) and bagging imputation to test dataset
df_test_complete <- df_test[complete.cases(df_test[,c("Temp3pm","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","WindGustSpeed")]),]
df_test_imputed <- predict(df_train_imp, newdata = df_test_complete)

# To check test dataset if those attributes still contain missing values or not nans_test_imp dataframe created
nans_test_imp=as.data.frame(matrix(ncol = 1, nrow = 18), stringsAsFactors = F)
for (i in 1:length(df_test_imputed)) {
  nans_test_imp[i,1] <- sum(is.na(df_test_imputed[,i]))/length(df_test_imputed[,i])
}

colnames(nans_test_imp) <- "Percentage of NA" # Assigning a column name
rownames(nans_test_imp) <- colnames(df_train) # Assigning row names
nans_test_imp <- nans_test_imp[order(nans_test_imp),,drop=F] #Sorted
nans_test_imp

#####BEFORE AND AFTER IMPUTATION#####
summary(df_train_complete)
summary(df_train_imputed)

summary(df_test_complete)
summary(df_test_imputed)

#####WRITE RESULTS TO CSV#####
df_train_imputed <- cbind(date_loc,df_train_imputed)
write.csv(df_train_imputed, file = "rain_from_R_train.csv")
write.csv(df_test_imputed, file = "rain_from_R_test.csv")



