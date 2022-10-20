import mnist_loader
training_data , validation_data , test_data = mnist_loader . load_data_wrapper ()


neurons_in_hidden_layer = 30
neurons_in_output  = 10
neurons_in_input = 784
max_hidden_layer = 20
max_rate = 10
obj = {}
activators = [sigmoid]
str_activators = ['sigmoid']
derivatives = [sigmoid_prime]
for rate in range(1,max_rate+1):
  obj[rate] = pd.DataFrame()
  obj[rate]["layer"] = [0]*max_hidden_layer
  obj[rate]["sigmoid"] = [0]*max_hidden_layer
  obj[rate]["tanh"] = [0]*max_hidden_layer
  obj[rate]["relu"] = [0]*max_hidden_layer
  obj[rate]["softmax"] = [0]*max_hidden_layer
  obj[rate]["relu"] = [0]*max_hidden_layer
  for layer in range(1,max_hidden_layer+1):
    obj[rate]["layer"][layer-1] = layer
    for ac in range(len(activators)):
      neurons = [neurons_in_input] + [neurons_in_hidden_layer]*layer + [neurons_in_output]
      training_data , validation_data , test_data = mnist_loader . load_data_wrapper ()
      net = Network(neurons,activators[ac],derivatives[ac])
      percentage = net.SGD(training_data , epochs, mini_batch_size, eta, test_data )
      obj[rate][str_activators][layer-1] = percentage
