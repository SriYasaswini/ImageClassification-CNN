# ImageClassification-CNN

Abstract: 
 
In order to tackle one of the most prominent challenges we face today – unhealthy diet, we are utilizing the technologies available
like Neural Networks, Image Augmentation and pairing them with vast amount of relevant data. 
The output is a model trained with Convolutional Neural Network which can identify 8 food items and then we give our user 
the nutritional information of their food scaled to 100 grams. 
 
 
 
Introduction: 
Problem: 
In today’s world as soon as people see something delicious, they jump to eat it without considering the excess cholesterol and calories which the food contains. 
 
 
Solution: 
with our application, the user will be able to fetch the nutritional value of the food item with just a snap of it. This will help them in making better dietary choices as diet is one of the primary constituents of good health and long life. 
 
 
Challenge:  
• One of the main challenges was training our models on the available dataset of 10GB worth of pictures.  • Applying Convolutional Neural Network with optimum learning parameters. 
 
 
Related Work: 
 • There are a few models which only identify one type of food.  
 • There are many image recognition projects which use CNN for classification. 
 
 
Methods: 
Pipeline of the project: 
 • Collection of the dataset.  
 • Training the model. 
 • Testing the model. 
 • Extending prediction of food with nutritional value. 
 
 
Packages: 
• Tensorflow 
 
 
 
Experiments: 
We conducted several experiments by tweaking the Batch Size, Epoch Number, Number of hidden layers and Number of neurons
in each hidden layer in the Neural Network.  
 
 
 
Results and discussion: 
After several iterations of experiments, the configuration which worked best regarding Accuracy and our computational capacity is – 
Batch Size = 128, Epoch = 5, Number of hidden layers = 3, Number of Neurons in each layer = [32, 32, 64]. 
 
 
 
Conclusion: 
 he model obtained can efficiently predict these 8 food items
 – 1. Chicken Curry 2. Chicken Wings 3. Pizza 4. Samosa 5. Sushi 6. Dumplings 7. Garlic Bread 8. French Fries 
 
With the nutritional value of the relevant food item displayed, the user will be able to decide whether to indulge in the delicacy or not. 
 
 
 
References: 
• Krizhevsky, Alex. “ImageNet Classification with Deep Convolutional Neural Networks”. papers.nips.cc/paper/4824
• Coleman, Robert. “tensorflow-image-classification”. github.com/rdcolema/tensorflow-imageclassification  
 
