
## Using transformers to classify text in R ##
## Created by Bronwen Hunter on 29/01/2022 ##


#install.packages('reticulate')
library(reticulate)
reticulate::install_miniconda(update=T, force=T) #This created a miniconda environment
conda_install('r-reticulate', c('pytorch', 'transformers', 'scikit-learn', 'tensorflow-gpu')) #Installing our python libraries
library(tensorflow)
install_tensorflow(version = "gpu") #This allows use to use the GPU

# Loading in the required R libraries
library(dplyr) 
library(plyr)
library(torch)

#install.packages('data.table')
#install.packages('text2vec')

# Now that we've installed the required python libraries into our environment
# we can load them in:

transformers <- reticulate::import('transformers') 
sklearn <- reticulate::import('sklearn')

# Checking a GPU is available and that it will be used:

physical_devices = tf$config$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(physical_devices[[1]],TRUE)
tf$keras$backend$set_floatx('float32')


# Here, I'm using a classic dataset available from text2vec
# Movie reviews are classified as positive or negative:
library(text2vec)
data("movie_review")
df <- movie_review %>%  sample_n(2000) %>% data.table::as.data.table()
df <- df %>% dplyr::rename(label_idx = sentiment, text=review)

idx_train = sample.int(nrow(df)*0.8) #Splitting the data into training and testing (80-20 split)

train = df[idx_train,]
test = df[!idx_train,]


##### Defining some base functions ####

# Function to set training arguments to be fed into Transformers trainer:

set_training_args <- function(output,              # Where models will be saved
                              no_epochs,           # Number of training epochs
                              batch_size,          # Training batch size
                              eval_batch_size){    # Evaluation batch size
  
  trainingargs <- transformers$TrainingArguments(
    output_dir=output,
    num_train_epochs=no_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size, 
    warmup_steps=500, 
    weight_decay=0.01,
    logging_dir=paste(output, '/logs'), 
    logging_steps=100,
    load_best_model_at_end=TRUE,
    evaluation_strategy="steps",
    seed = 42)
  
  return(trainingargs)
}

# This function sets up the transformer training module
setup_trainer <- function(model, training_args, train_dataset, eval_dataset) {
  trainer <- transformers$trainer(
    model=model,
    args= training_args,
    train_dataset = train_dataset,
    eval_dataset=eval_dataset,          
    compute_metrics=compute_metrics)
  return(trainer)
}

# Function to compute the performance of classifiers after training
compute_metrics <- function(pred){
  labels <- pred$labelids
  preds <- pred$predictions$argmax(-1)
  target_names <- list('Relevant', 'Irrelevant')
  mylist <- list(sklearn$metrics$precision_recall_fscore_support(labels, preds, average='binary'))
  precision <-list[1]
  recall <-list[2]
  f1 <- list[3]
  support <-list[4]
  accuracy <- sklearn$metrics$accuracy_score(labels, preds)
  report <- sklearn$metrics$classification_report(abels, preds, target_names=target_names)
  return(list('precision'=precision, 'recall'=recall, 'f1'=f1, 'report'=report, 'accuracy'=accuracy))
}

# Function to train the classifier
train_model <- function(trainer){
  print("Starting training")
  training <- trainer$train()
  print(training)
  print('training finished')
  mylist <- list('trainer'=trainer, 'training' = training)
  return(mylist)
  
}

# Function to evaluate the classifier
evalute_model <- function(trainer){
  evaluation <- trainer$evaluate()
  print("Staring evaluation")
  print(evaluation)
  print("Finished evaluation")
  return(evaluation)
}


#Here we set the maximum length of texts (max length fpr Bert is 512):
max_length<- as.integer(50)

# Loading in the tokenizer:
tokenizer <- transformers$BertTokenizer$from_pretrained('bert-base-uncased')

# Loading in the classifier model
model <- transformers$BertForSequenceClassification$from_pretrained('bert-base-uncased') 

# This function allows use to get the sequence encodings from the tokenizer:
get_encodings <- function(data){for (i in 1:nrow(data)) {
  text_ids = list()
  text_mask = list()
  txt = tokenizer$encode_plus(data[['text']][i], max_length=100L) 
  text_ids = text_ids %>% append(txt$input_ids %>% t() %>% as.matrix() %>% list())
  text_mask = text_mask %>% append(txt$token_type_ids  %>% t() %>% as.matrix() %>% list())}
  return(list('input_ids'=text_ids, 'masked'=text_mask))
}


train_encodings = get_encodings(train)
test_encodings = get_encodings(test)

# Now we need to convert this into a torch tensor
train_seq <- torch_tensor(unlist(train_encodings[['input_ids']])) #These are the id numbers of each token
train_mask <- torch_tensor(unlist(train_encodings[['token_type_ids']]))
train_y <- torch_tensor(unlist(train$label_idx))

test_seq <- torch_tensor(unlist(test_encodings[['input_ids']]))
test_mask <- torch_tensor(unlist(test_encodings[['token_type_ids']]))
test_y <- torch_tensor(unlist(test_label)) 

# Now wrapping these in a tensor dataset

train_dataset <- tensor_dataset(train_seq, train_mask, train_y)
test_dataset <- tensor_dataset(test_seq, test_mask, test_y)

data_train <- dataloader(train_dataset, batch_size = 32)
data_test <- dataloader(test_dataet, batch_size=32)

# Now we are ready to train and evaluate!

training_args <- set_training_args('C:/Users/bronn/Documents/R Scripts/Models',2,30,30)
trainer <- setup_trainer(model, training_args, data_train, data_test)
trainer_trained <- train_model(trainer)
evaluation <- evaluate_model(trainer_trained)

#Now we can use the trainer to make predictions about the class of other texts

trainer_trained$predict(text)

transformers$Trainer()




