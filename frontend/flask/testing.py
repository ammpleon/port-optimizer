import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad = False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad = False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad = False)
        self.b10 = nn.Parameter(torch.tensor(0.00), requires_grad = False)
        self.w11 = nn.Parameter(torch.tensor(2.70), requires_grad = False)

        self.final_bias = nn.Parameter(torch.tensor(0), requires_grad = False)

    def forward(self, input):

        inputToTopRelu = input * self.w00 + self.b00
        topReluScaled = F.relu(inputToTopRelu) * self.w01

        inputToBottomRelu = input * self.w10 + self.b10
        bottomReluScaled = F.relu(inputToBottomRelu) * self.w11

        inputToFinalRelu = topReluScaled + bottomReluScaled + self.final_bias
        output = F.relu(inputToFinalRelu)
        return output
    


class BasicNN_train(nn.Module):

    def __init__(self,final_bias = 0.):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad = False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad = False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad = False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad = False)
        self.b10 = nn.Parameter(torch.tensor(0.00), requires_grad = False)
        self.w11 = nn.Parameter(torch.tensor(2.70), requires_grad = False)

        self.final_bias = nn.Parameter(torch.tensor(final_bias), requires_grad = True)

    def forward(self, input):

        inputToTopRelu = input * self.w00 + self.b00
        topReluScaled = F.relu(inputToTopRelu) * self.w01

        inputToBottomRelu = input * self.w10 + self.b10
        bottomReluScaled = F.relu(inputToBottomRelu) * self.w11

        inputToFinalRelu = topReluScaled + bottomReluScaled + self.final_bias
        output = F.relu(inputToFinalRelu)
        return output
    



input_doses = torch.linspace(start = 0, end = 1, steps = 11)

# model = BasicNN()

# output_values = model(input_doses)

# sns.set(style = "whitegrid")
# sns.lineplot(x = input_doses,
#              y = output_values,
#              color = "green",
#              linewidth = 2.5)
# plt.title("Dosages Effectiveness")
# plt.xlabel("effectiveness")
# plt.ylabel("Dose")

# plt.show()


model_train = BasicNN_train()
output_vals = model_train(input_doses)

sns.set(style = "whitegrid")
sns.lineplot(x = input_doses,
             y = output_vals.detach(),
             color = "green",
             linewidth = 2.5)
plt.xlabel("Dosage")
plt.ylabel("Effectiveness")






#backpropagation
optimizer = SGD(model_train.parameters(), lr = 0.1) #stochastic gradient descent
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
print(f"Bias before optimization: {str(model_train.final_bias.data)}\n")

plt.show() #plot before bias optimization

for epoch in range(100):

    total_loss = 0

    for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]

        output_i = model_train(input_i)

        loss = F.mse_loss(output_i, label_i)

        loss.backward()
        total_loss += float(loss)

    if (total_loss < 0.0001):
        print(f"Num steps {str(epoch)}")
        break

    else:
        optimizer.step()
        optimizer.zero_grad()

        print(f"step: {str(epoch)}\nFinal Bias: {str(model_train.final_bias.data)}\n")

print(f"Final bias after optimization: {model_train.final_bias.data}")

final_bias = float(model_train.final_bias.data)

model_test = BasicNN_train(final_bias = final_bias)
outputs = model_test(input_doses)

sns.set(style = "whitegrid")
sns.lineplot(x = input_doses,
             y = outputs.detach(),
             color = "green",
             linewidth = 2.5)
plt.xlabel("Dosage")
plt.ylabel("effectiveness")


plt.show() #plot after bias optimization





