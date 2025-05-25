# loading the package
library(keras3)

# loading the data set
boston <- dataset_boston_housing()

# separating the data set in train data and test data
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

# normalizing the data
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center = mean, scale = sd)

# building the model
build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1)
  model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metrics = "mae")
  model
}

# creating cross validation data
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 100
all_scores <- numeric()


# runing the model in each part of the cross validation data
for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  val_indices <- which(fold_id == i)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit (
                 partial_train_data,
                 partial_train_targets,
                 epochs = num_epochs,
                 batch_size = 16,
                 verbose = 0
  )
  
  results <- model %>%
    evaluate(val_data, val_targets, verbose = 0)
  all_scores[[i]] <- results[['mae']]
}

# showing the scores of the mae and the mean of all of them
all_scores
mean(all_scores)

# runing the model in each part of the cross validation data but for biger size of epoch
num_epochs <- 500
all_mae_histories <- list()
for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  val_indices <- which(fold_id == i)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  model <- build_model()
  history <- model %>% fit(
                           partial_train_data, partial_train_targets,
                           validation_data = list(val_data, val_targets),
                           epochs = num_epochs, batch_size = 16, verbose = 0
  )
  mae_history <- history$metrics$val_mae
  all_mae_histories[[i]] <- mae_history
}

# visualizing how much the MAE decreases
all_mae_histories <- do.call(cbind, all_mae_histories)
average_mae_history <- rowMeans(all_mae_histories)
plot(average_mae_history, xlab = "epoch", type = 'l')


# training the model with all the data set
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 120, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)

# showing the mae of the final model
result["mae"]


# predicting the price of the house 
price_predictions <- model %>% predict(test_data)

