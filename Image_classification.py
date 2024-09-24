import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import ResNet18_Weights

if __name__ == '__main__':

    # Defining the data transformations for data augmentation and normalization
    # The reason this is done is for our model to generalize the data it is fed, so that it learns faster
    data_transforms = {
        'Train': transforms.Compose([
            # Crops the images randomly to 224 size
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # MANDATORY: Since pytorch accepts the data in the form of a tensor
            transforms.ToTensor(),
            # Normalize the scale of all images to the same scale
            # Uses mean and standard deviation behind the scenes
            # Normalizing the data allows the CNN to learn faster by reducing the skewness
            # [R, G, B]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Val': transforms.Compose([
            # Crops the images randomly to 224 size
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # MANDATORY: Since pytorch accepts the data in the form of a tensor
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Define the directory of our dataset
    data_dir = 'Dataset'

    # Create data loaders
    # Data loaders load the training and validation data to our model to use
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Val']}

    # image_datasets
    # Data is loaded in mini batches of 4, data will keep shuffling, num_workers = 4 means 4 processes can occur at once (parallel processing)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['Train', 'Val']}
    # dataset_sizes returns the number of training records in both the training and validation folders, useful for quickly seeing how many
    # training records I have and whether to increase it over time
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Val']}
    print(dataset_sizes)

    # Now we want to store the folder names the same way as the dataset_sizes above, this just outputs the two folders I have in the Train set
    class_names = image_datasets['Train'].classes
    print(class_names)

    # Loading the pretrained model "Resnet18"
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers except final classification (AKA fully connected) layer
    for name, param in model.named_parameters():
        # if the name is "fc" which means fully connected (final layer) then THAT layer WILL be subject to training
        # because we WANT to use that final layer
        if "fc" in name:
            param.requires_grad = True
        # Otherwise, any other layer is not subject to training and will be "frozen"
        else:
            param.requires_grad = False

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # If the GPU is available, use it for the model training else use the cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # training
    # an epoch is one complete pass of the training dataset through the algorithm, since I have a small number of datapoints, only 10 epochs are used
    num_epochs = 10
    for epoch in range(num_epochs):
        # If the phase is "train" then we want to train the model if it's not, then we want to evaluate the model
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            # These keep a running tally of the losses and correct guesses in each epoch
            running_loss = 0.0
            running_corrects = 0

            # inputs are the images, labels are the output labels (the desired class name like "cute" or "silly"
            # Then it sends it to the device defined on line 69 (cpu or GPU)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # This resets the gradient from the previous epoch iterations
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    # The outputs will be the predictions of the model when input images are fed into it
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # This loss is assigned the value of the cross-entropy loss between the outputs (the predictions) and
                    # The real value of the label that the image should have (the classification)
                    loss = criterion(outputs, labels)
                    # CNN uses backward and forward propagation in a continuous cycle (forward -> backward -> forward-> ...)
                    # First we use forward propagation (provide input to the model, model predicts the corresponding label)
                    # Next back propagation (gradient is calculated, weights and biases are updated)
                    if phase == "Train":
                        # Gradient calculated
                        loss.backward()
                        # Weights updated using calculated gradient
                        optimizer.step()
                # Running loss and corrects are updated
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # These two values show the accuracy and loss PER EPOCH (ideally over the course of the training the loses should decrease)
            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')
    print("Training complete")
    # Save the model to be used on an unseen image
    torch.save(model.state_dict(), 'laban_classification_model.pth')
    '''THIS PART BELLOW SHOULD BE COMMENTED OUT UNTIL ALL THE TRAINING AND VALIDATION IMAGES ARE TRAINED INTO THE MODEL'''
    # Use the model to predict unseen images
    # Load the saved model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features,
                         1000)  # The original resnet18 pretrained model has 1000 classes (labels) so we need to match that
    model.load_state_dict(torch.load('laban_classification_model.pth'))
    model.eval()

    # Create a new model using the correct final layer because we are only using that layer from the resnet model, we froze all the others
    new_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    new_model.fc = nn.Linear(new_model.fc.in_features,
                             2)  # Since our model only uses 2 different classes, we specify that instead of using all 1000

    # Copy the weights and biases from the loaded model into the new model
    new_model.fc.weight.data = model.fc.weight.data[0:2]  # Only copying the first 2 output units
    new_model.fc.bias.data = model.fc.bias.data[0:2]


    # Load and prepare the unseen image
    image_path = 'test_cute_laban_silly_harder_jpg.jpg'  # This path should match exactly with the test image path
    image = Image.open(image_path)
    # Repeat the processing steps that were done for the training and validation images at the beginning
    preprocess = transforms.Compose([
        # Crops the images randomly to 224 size
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # MANDATORY: Since pytorch accepts the data in the form of a tensor
        transforms.ToTensor(),
        # Normalize the scale of all images to the same scale
        # Uses mean and standard deviation behind the scenes
        # Normalizing the data allows the CNN to learn faster by reducing the skewness
        # [R, G, B]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform the prediction
    with torch.no_grad():
        output = model(input_batch)
    # Get the predicted class
    _, predicted_class = output.max(1)

    # Map the predicted class to the class name
    class_names = ['Cute', 'Silly']
    predicted_class_name = class_names[predicted_class.item()]

    print(f'The predicted class is: {predicted_class_name}')

    # Display the image with a label on it using matplotlib
    image = np.array(image)
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
    plt.show()
''''''
