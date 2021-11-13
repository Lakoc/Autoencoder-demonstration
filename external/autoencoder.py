import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from external.Network import Network

INPUT_SIZE = 2


class AE(nn.Module):
    def __init__(self, INPUT_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, INPUT_SIZE),
            nn.Tanh(), )

    def forward(self, x):
        f_2 = self.encoder(x)
        x = self.decoder(f_2)
        return x, f_2


if __name__ == "__main__":
    epochs = 5000
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(INPUT_SIZE).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()
    N = 100
    mean1 = torch.Tensor([[0.5, 0]])
    mean2 = torch.Tensor([[0, 0.5]])
    # mean3 = torch.Tensor([[0, 0.2, 0.5, -0.6]])
    # mean4 = torch.Tensor([[0, 0.5, 0, 0.7]])
    cov = torch.eye(INPUT_SIZE) / 100
    distrib1 = MultivariateNormal(loc=mean1, covariance_matrix=cov)
    distrib2 = MultivariateNormal(loc=mean2, covariance_matrix=cov)
    # distrib3 = MultivariateNormal(loc=mean3, covariance_matrix=cov)
    # distrib4 = MultivariateNormal(loc=mean4, covariance_matrix=cov)
    x1 = torch.squeeze(distrib1.sample_n(N))
    x2 = torch.squeeze(distrib2.sample_n(N))
    # x3 = torch.squeeze(distrib3.sample_n(N))
    # x4 = torch.squeeze(distrib4.sample_n(N))
    batch_features = torch.cat((x1, x2), 0).detach().numpy()
    batch_features= batch_features[:,  :, np.newaxis]

    net = Network([2, 3, 2])
    training_data = [(val, val) for val in batch_features]

    net.SGD(training_data, 1000, 10, 1.0)

    x = 5

    # baeatures /= max(torch.max(batch_features), np.abs(torch.min(batch_features)))
    # num_epochs = 100
    # # do = nn.Dropout()
    # # comment out for under AE
    # features = np.zeros((4 * N, 2))
    # for epoch in range(num_epochs):
    #     for index, data in enumerate(batch_features):
    #         # ************************ forward *************************
    #         output, feature = model(data)  # uncomment for fully connected
    #         features[index] = output.detach().numpy()
    #         # feed img_bad for over AE
    #         # output = conv_model(img)                       # uncomment for convulation AE
    #         loss = criterion(output, data)
    #         # ************************ backward *************************
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # ***************************** log ***************************
    #     print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}')
    #     if epoch % 10 == 0:
    #         plt.plot(features[0:N, 0], features[0:N, 1], 'ro')
    #         plt.plot(features[N:2 * N, 0], features[N:2 * N, 1], 'bo')
    #         plt.plot(features[2 * N:3 * N, 0], features[2 * N:3 * N, 1], 'go')
    #         plt.plot(features[3 * N:, 0], features[3 * N:, 1], 'o')
    #         plt.show()
    # for epoch in range(epochs):
    #     train_loss = 0
    #     # reset the gradients back to zero
    #     # PyTorch accumulates gradients on subsequent backward passes
    #     optimizer.zero_grad()
    #
    #     # compute reconstructions
    #     outputs, features = model(batch_features)
    #     if epoch % 10 == 0:
    #         f_p = features.detach().numpy()
    #         plt.plot(f_p[0:2, 0], f_p[0:2, 1], 'ro')
    #         plt.plot(f_p[2:, 0], f_p[2:, 1], 'bo')
    #         # plt.plot(f_p[6:, 0], f_p[6:, 1], 'go')
    #         plt.show()
    #     # compute training reconstruction loss
    #     train_loss += torch.sum(torch.square(features[0] - features[2]))
    #     train_loss += torch.sum(torch.square(features[0] - features[3]))
    #     train_loss += torch.sum(torch.square(features[1] - features[2]))
    #     train_loss += torch.sum(torch.square(features[1] - features[3]))
    #     train_loss = 100 - train_loss
    #
    #     # compute accumulated gradients
    #     train_loss.backward()
    #
    #     # perform parameter update based on current gradients
    #     optimizer.step()
    #
    #     # add the mini-batch training loss to epoch loss
    #     print(train_loss)
