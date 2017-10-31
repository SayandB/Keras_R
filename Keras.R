library(keras)

#Loading MNIST data, comes alongwith the package
data <- dataset_mnist()
train <- data$train
test <- data$test
Xtrain <- train$x
Ytrain <- train$y
Xtest <- test$x
Ytest <- test$y
rm(data)
rm(train)
rm(test)

#Reshaping data and normalisation
Xtrain <- array(Xtrain, dim = c(dim(Xtrain)[1], prod(dim(Xtrain)[-1])))/255
Xtest <- array(Xtest, dim = c(dim(Xtest)[1], prod(dim(Xtest)[-1])))/255

#One-hot encoding!
Ytrain <- to_categorical(Ytrain, 10)
Ytest <- to_categorical(Ytest, 10)

#Building model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 784, input_shape = 784, use_bias = TRUE) %>%
  layer_dropout(rate = 0.25) %>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10, use_bias = TRUE) %>%
  layer_activation(activation = 'softmax')
model %>% compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = c('accuracy'))
model %>% fit(Xtrain, Ytrain, epochs = 500, batch_size = 100)

#Model performance and prediction
result <- model %>% evaluate(Xtest, Ytest, batch_size = 100)
prediction <- model %>% predict_classes(Xtest)

