import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        # use densenet121
        self.densenet = models.densenet121(pretrained=True)
        
        # replace densenet classification layer with fully conected linear layer
        self.densenet.classifier = nn.Linear(1024, out_features=512)
        
        # add fully conected embedding layer
        self.embed = nn.Linear(512, embed_size)
        
        # add dropout regularization
        self.dropout = nn.Dropout(p=0.5)
        
        #add relu activation for non-linarity
        self.prelu = nn.PReLU()

    def forward(self, images):
        
        features = self.dropout(self.prelu(self.densenet(images)))
        
        features = self.embed(features)
        
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        
        self.hidden_size = hidden_size
        
        self.vocab_size = vocab_size
        
        self.num_layers = num_layers
        
        # add lstm cell
        self.lstm = nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.hidden_size, 
                            num_layers = self.num_layers, 
                            batch_first=True)
    
        # add output fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # add embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
    def forward(self, features, captions, teacher_forcer = False):
        
        if teacher_forcer:
            """ Implementation with teacher forcer did not provide meaningfully improved results."""
            # defining output tensor
            batch_size = features.size(0)
            out = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
        
            embeded = self.embed(captions)
        
            # loop over caption, use teacher forcer past step 0
            for t in rage(captions.size(1)):
                if t == 0:
                    ltsm_output, states = self.lstm(features.unsqueeze(1))
                else:
                    ltsm_output, states = self.lstm(embeded[:, t, :], states)
        
                output = self.fc(lstm_output)
                out[:,t,:] = output
        else:
        
            # embed captions without <end> token
            embeded = self.embed(captions[:,:-1])
        
            # combine feature and embeded captions
            embeded = torch.cat((features.unsqueeze(1), embeded), dim = 1)
        
            # feed to the lstm
            lstm_out, _ = self.lstm(embeded)
        
            out = self.fc(lstm_out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # initialize states
        states = (torch.randn(1, 1 , self.hidden_size).to(inputs.device), 
                  torch.randn(1, 1 , self.hidden_size).to(inputs.device))
        
        # create empty output list
        output = []
        
        # loop through the prediction
        for i in range(max_len):

            prediction, states = self.lstm(inputs, states)
            
            # most likely prediction
            prediction = self.fc(prediction)
            
            _, prediction = torch.max(prediction,2)
            
            # build the output tensor
            output.append(prediction.item())
         
            # feed the prediction into the LSTM for the next token
            inputs = self.embed(prediction)
    
        return output