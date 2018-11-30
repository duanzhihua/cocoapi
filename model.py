import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size #512
        self.num_layers = num_layers #2
        self.embed = nn.Embedding(vocab_size, embed_size) # (8856,256)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size) #(512 ,8856)
    
    def forward(self, features, captions):
        """
        :param features: shape is (bs, embed_size)
        :param captions: shape is (bs, length) [1, 13]
        :return:
        """
        cap_embed = self.embed(captions[:, :-1])  #[1, 12] (bs, len, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), cap_embed), dim=1) # [1,1,256] ,[1,12,256]--->[1, 13,256] # (bs, lens, embed_size)
        outputs, _ = self.rnn(embeddings)  #[1,13,512] (bs, lens, hidden_size)
        scores = self.fc(outputs)  # [1, 13, 8856] (bs, lens, vocab_size)
        return scores

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted
        sentence (list of tensor ids of length max_len)
        :param inputs: shape is (1, 1, embed_size)
        """
        if states == None:
            states = (torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device),
                      torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device))
        outputs = list()
        for i in range(max_len):
            scores, states = self.rnn(inputs, states)  # scores: (1, 1, vocab_size)
            scores = self.fc(scores.squeeze(1))  # (1, vocab_size)
            output = scores.max(1)[1]
            outputs.append(output.item())
            inputs = self.embed(output.unsqueeze(1))  # (1, 1, embed_size)
        return outputs